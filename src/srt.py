from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from netket import jax as nkjax
from netket import config
from netket.utils import timing
from netket.utils.types import Array
from netket import stats
import platform

from .mesh import create_2d_mesh

system = platform.system()

if system == "Linux":
    from jaxmg import potrs


def streamed_gram(O_L, chunk_size):
    ndev = jax.device_count()
    shard_size, n_params = O_L.shape
    padding = (-n_params) % chunk_size
    if padding:
        print(f"WARNING: padding by {padding}, will add double memory pressure")
        O_L = jnp.pad(O_L, pad_width=((0, 0), (0, padding)), mode="constant")
    n_params_padded = O_L.shape[1]
    n_chunks = n_params_padded // chunk_size
    O_L = jnp.reshape(O_L, (shard_size, n_chunks, n_params_padded // n_chunks))
    O_L = jnp.transpose(O_L, (1, 0, 2))

    def step(_b, Oc_loc):
        Oc_all = jax.lax.all_gather(Oc_loc, ('S', 'T'), axis=0, tiled=True)  # tiled: (ndev*shard_size, chunk)
        _b += Oc_loc @ Oc_all.T  # (shard_size, chunk) @ (chunk, ndev*shard_size) = (shard_size, ndev*shard_size)
        return _b, None
    # Accumulator: mark as pvary since all_gather with tiled=True introduces varying axes
    block0 = jax.lax.pvary(jnp.zeros((shard_size, ndev * shard_size), dtype=O_L.dtype), axis_name=("S", "T"))
    block, _ = jax.lax.scan(step, block0, O_L)
    return block


def update_O_LT(O_LT, x):
    return jax.lax.psum(x @ O_LT, axis_name=("S", "T"))

@jax.jit
def shift(A, diag_shift):
    idx = jnp.arange(A.shape[0])
    return A.at[idx, idx].add(diag_shift)

# @jax.jit
def shift_shardmap(A, diag_shift):
    shard_size = A.shape[0]
    idx = jnp.arange(A.shape[0])
    A = A.at[idx, idx + shard_size * jax.lax.axis_index(("S", "T"))].add(diag_shift)
    return A

@timing.timed
@partial(
    jax.jit,
    static_argnames=(
            "solver_fn",
            "mode",
            "chunk_size",
    ),
)
def _compute_srt_update(
        O_L,
        dv,
        *,
        chunk_size: int = None,
        diag_shift: float | Array,
        solver_fn: Callable[[Array, Array], Array],
        mode: str,
        params_structure,
):
    # (#ns, np) -> (ns, #np)
    O_LT = O_L
    mesh_2d = create_2d_mesh()
    # jax.sharding.set_mesh(mesh_2d)
    if config.netket_experimental_sharding:
        nkjax.sharding.pad_axis_for_sharding(O_LT, axis=1, padding_value=0.0)
        O_LT = jax.lax.with_sharding_constraint(
            O_LT,
            NamedSharding(mesh_2d, P(("S", "T"), None)),
        )
        dv = jax.lax.with_sharding_constraint(
            dv, NamedSharding(mesh_2d, P())
        )

    # This does the contraction (ns, #np) x (#np, ns) -> (ns, ns).
    # When using sharding the sum over #ns is done automatically.
    # When using MPI we need to do it manually with an allreduce_sum.
    if chunk_size is not None:
        matrix = jax.shard_map(partial(streamed_gram, chunk_size=chunk_size),
                       mesh=mesh_2d, in_specs=P(("S", "T"), None), out_specs=P(("S", "T"), None))(O_LT)
    else:
        matrix = O_LT @ O_LT.T
   
    shifted_matrix = shift(matrix, diag_shift)

    shifted_matrix = jax.lax.with_sharding_constraint(shifted_matrix, P("T", None))
    
    if system == "Darwin":
        aus_vector = solver_fn(shifted_matrix, dv)
    elif system == "Linux":
        aus_vector = potrs(shifted_matrix, dv[:, None], T_A=2048, mesh=mesh_2d,
                                    in_specs=(P("T", None), P(None, None)), return_status=True)
    # Some solvers return a tuple, some others do not.
    if isinstance(aus_vector, tuple):
        aus_vector, info = aus_vector
        if info is None:
            info = {}
    else:
        info = {}
    aus_vector = aus_vector.squeeze()
    # aus_vector.block_until_ready()
    jax.debug.print("info {}", info)
    # (np, #ns) x (#ns) -> (np).
    updates = jax.shard_map(update_O_LT,
                            mesh=mesh_2d,
                            in_specs=(P(("S", "T"), None), P(("S", "T"))),
                            out_specs=P(None))(O_LT, aus_vector)
    # assert jnp.allclose(updates, updates_old)
    # If complex mode and we have complex parameters, we need
    # To repack the real coefficients in order to get complex updates
    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        num_p = updates.shape[-1] // 2
        updates = updates[:num_p] + 1j * updates[num_p:]

    if config.netket_experimental_sharding:
        out_shardings = NamedSharding(
            mesh_2d, P(*(None,) * updates.ndim)
        )
        updates = jax.lax.with_sharding_constraint(updates, out_shardings)

    return updates, info

@timing.timed
# @jax.jit
def _compute_snr(
        O_L,
        dv,
        *,
        diag_shift: float | Array,
):
    # (#ns, np) -> (ns, #np)
    O_LT = O_L
    # mesh_2d = create_2d_mesh()
    # jax.sharding.set_mesh(mesh_2d)
    if config.netket_experimental_sharding:
        nkjax.sharding.pad_axis_for_sharding(O_LT, axis=1, padding_value=0.0)
        O_LT = jax.lax.with_sharding_constraint(
            O_LT,
            NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None)),
        )
        dv = jax.lax.with_sharding_constraint(
            dv, NamedSharding(jax.sharding.get_abstract_mesh(), P())
        )

    # This does the contraction (ns, #np) x (#np, ns) -> (ns, ns).
    # When using sharding the sum over #ns is done automatically.
    # When using MPI we need to do it manually with an allreduce_sum.
    matrix = O_LT @ O_LT.T
    # matrix_side = matrix.shape[-1]  # * it can be ns or 2*ns, depending on mode
    # shifted_matrix = jax.lax.add(
    #     matrix, diag_shift * jnp.eye(matrix_side, dtype=matrix.dtype)
    # )
    shifted_matrix = shift(matrix, diag_shift)
    shifted_matrix = shift(matrix, diag_shift)
    # replicate
    shifted_matrix = shifted_matrix.T
    # shifted_matrix = jax.lax.with_sharding_constraint(shifted_matrix, NamedSharding(jax.sharding.get_abstract_mesh(), P(None, "S")))

    if system == "Darwin":
        ev, V = jnp.linalg.eigh(shifted_matrix)
    elif system == "Linux":

        jax.debug.print("starting solver")
        # ev, V = syevd(shifted_matrix, T_A=256, mesh=mesh_2d,
        #               in_specs=(P(None, "S"),))
        ev, V = jnp.linalg.eigh(shifted_matrix)
        jax.debug.print("finished solver")

    n_samples = O_L.shape[0]
    # V.T Sigma V = O_LT * (Eloc - E)
    OEdata = O_LT * jnp.expand_dims(dv, axis=-1)  # /N_mc
    # Sigma V = V @ (O_LT * (Eloc - E))
    QEdata = V.T @ OEdata  # /N_mc

    rho = stats.mean(QEdata, axis=0)  # /sqrt(N_mc)
    # Compute the SNR according to Eq. 21 but taking care of where sigma_k is zero
    sigma_k = jnp.maximum(jnp.sqrt(stats.var(QEdata, axis=0)), 1e-14)
    # Here we are hardcoding the case where rho==0 and sigma_k==0 to have infinite snr.
    # This is an arbitrary choice, but avoids generating NaNs in the snr calculation.
    # See netket#1959 and #1960 for more details.
    snr = jnp.where(
        sigma_k == 0,
        jnp.inf,
        jnp.abs(rho) * jnp.sqrt(n_samples) / sigma_k,
    )
    # jax.sharding.set_mesh(Mesh(jax.devices(), ("S")))
    return snr
