from collections.abc import Callable
from functools import partial

from einops import rearrange

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from jax.sharding import NamedSharding, PartitionSpec as P

from netket import jax as nkjax
from netket import config
from netket.jax._jacobian.default_mode import JacobianMode
from netket.utils import timing
from netket.utils.types import Array

from netket.jax import _ntk as nt


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "log_psi",
        "solver_fn",
        "chunk_size",
        "mode",
    ),
)
def srt_onthefly(
    log_psi,
    local_energies,
    parameters,
    model_state,
    samples,
    *,
    diag_shift: float | Array,
    solver_fn: Callable[[Array, Array], Array],
    mode: JacobianMode,
    proj_reg: float | Array | None = None,
    momentum: float | Array | None = None,
    old_updates: Array | None = None,
    chunk_size: int | None = None,
    pdf: Array,
):
    N_mc = local_energies.size
    if pdf is not None:
        pdf = pdf / N_mc        # normalize to probability distribution: sum=1
    else:
        pdf = jnp.ones(N_mc) / N_mc       # uniform distribution
    # Split all parameters into real and imaginary parts separately
    parameters_real, rss = nkjax.tree_to_real(parameters)

    # complex: (Nmc) -> (Nmc,2) - splitting real and imaginary output like 2 classes
    # real:    (Nmc) -> (Nmc,)  - no splitting
    def _apply_fn(parameters_real, samples):
        variables = {"params": rss(parameters_real), **model_state}
        log_amp = log_psi(variables, samples)

        if mode == "complex":
            re, im = log_amp.real, log_amp.imag
            return jnp.concatenate(
                (re[:, None], im[:, None]), axis=-1
            )  # shape [N_mc,2]
        else:
            return log_amp.real  # shape [N_mc, ]

    def jvp_f_chunk(parameters, vector, samples):
        r"""
        Creates the jvp of the function `_apply_fn` with respect to the parameters.
        This jvp is then evaluated in chunks of `chunk_size` samples.
        """
        f = lambda params: _apply_fn(params, samples)
        _, acc = jax.jvp(f, (parameters,), (vector,))
        return acc

    # compute rhs of the linear system
    local_energies = local_energies.flatten()
    de = local_energies - jnp.sum(pdf * local_energies)   # weighted energy centering

    # RHS of the kernel system: b_s = 2 * sqrt(p_s) * Delta_E_s
    # For uniform p_s=1/N this reduces to 2*de/sqrt(N).
    dv = 2.0 * jnp.sqrt(pdf) * de  # shape [N_mc,]
    if mode == "complex":
        dv = jnp.stack([jnp.real(dv), jnp.imag(dv)], axis=-1)  # shape [N_mc,2]
    else:
        dv = jnp.real(dv)  # shape [N_mc,]

    if momentum is not None:
        if old_updates is None:
            old_updates = tree_map(jnp.zeros_like, parameters_real)
        else:
            acc = nkjax.apply_chunked(
                jvp_f_chunk, in_axes=(None, None, 0), chunk_size=chunk_size
            )(parameters_real, old_updates, samples)

            _pdf = pdf[:, None] if mode == "complex" else pdf
            avg = jnp.sum(_pdf * acc, axis=0)   # weighted mean
            acc = jnp.sqrt(_pdf) * (acc - avg)  # weighted centering + sqrt(p) scale
            dv -= momentum * acc

    if mode == "complex":
        dv = jax.lax.collapse(dv, 0, 2)  # shape [2*N_mc,] or [N_mc, ] if not complex

    # Collect all samples on all MPI ranks, those label the columns of the T matrix
    all_samples = samples
    if config.netket_experimental_sharding:
        samples = jax.lax.with_sharding_constraint(
            samples, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
        )
        all_samples = jax.lax.with_sharding_constraint(
            samples, NamedSharding(jax.sharding.get_abstract_mesh(), P())
        )

    _jacobian_contraction = nt.empirical_ntk_by_jacobian(
        f=_apply_fn,
        trace_axes=(),
        vmap_axes=0,
    )

    def jacobian_contraction(samples, all_samples, parameters_real):
        if config.netket_experimental_sharding:
            parameters_real = jax.lax.pvary(parameters_real, "S")
        if chunk_size is None:
            # STRUCTURED_DERIVATIVES returns a complex array, but the imaginary part is zero
            # shape [N_mc/p.size, N_mc, 2, 2]
            return _jacobian_contraction(samples, all_samples, parameters_real).real
        else:
            _all_samples, _ = nkjax.chunk(all_samples, chunk_size=chunk_size)
            ntk_local = jax.lax.map(
                lambda batch_lattice: _jacobian_contraction(
                    samples, batch_lattice, parameters_real
                ).real,
                _all_samples,
            )
            if mode == "complex":
                return rearrange(ntk_local, "nbatches i j z w -> i (nbatches j) z w")
            else:
                return rearrange(ntk_local, "nbatches i j -> i (nbatches j)")

    # If we are sharding, use shard_map manually
    if config.netket_experimental_sharding:
        mesh = jax.sharding.get_abstract_mesh()
        # SAMPLES, ALL_SAMPLES PARAMETERS_REAL
        in_specs = (P("S", None), P(), P())
        out_specs = P("S", None)

        # By default, I'm not sure whether the jacobian_contraction of NeuralTangents
        # Is correctly automatically sharded across devices. So we force it to be
        # sharded with shard map to be sure

        jacobian_contraction = jax.shard_map(
            jacobian_contraction,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
        )

    # This disables the nkjax.sharding_decorator in here, which might appear
    # in the apply function inside.
    with nkjax.sharding._increase_SHARD_MAP_STACK_LEVEL():
        ntk_local = jacobian_contraction(samples, all_samples, parameters_real).real

    # shape [N_mc, N_mc, 2, 2] or [N_mc, N_mc]
    if config.netket_experimental_sharding:
        ntk = jax.lax.with_sharding_constraint(
            ntk_local, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
        )
    else:
        ntk = ntk_local
    # print(ntk.shape)
    w_sqrt = jnp.sqrt(pdf)  # [N_mc]

    if mode == "complex":
        # shape [2*N_mc, 2*N_mc] checked with direct calculation of J^T J
        ntk = rearrange(ntk, "i j z w -> (i z) (j w)")
        ntk = ntk.reshape(N_mc, 2, N_mc, 2)
        # Weighted double-centering on sample axes (0 and 2).
        # m_col[i,a,b] = sum_j p_j * K[i,a,j,b]  (mean over j, axis 2)
        # m_row[a,j,b] = sum_i p_i * K[i,a,j,b]  (mean over i, axis 0)
        m_col = jnp.einsum("iajb,j->iab", ntk, pdf)      # [N_mc, 2, 2]
        m_row = jnp.einsum("iajb,i->ajb", ntk, pdf)      # [2, N_mc, 2]
        m_global = jnp.einsum("i,iab->ab", pdf, m_col)   # [2, 2]
        ntk = (ntk
               - m_col[:, :, None, :]        # [N_mc, 2,  1,  2]
               - m_row[None, :, :, :]        # [ 1,  2, N_mc, 2]
               + m_global[None, :, None, :]) # [ 1,  2,  1,  2]
        # Apply W^{1/2}: tilde_K[i,a,j,b] = sqrt(p_i)*sqrt(p_j) * C[i,a,j,b]
        ntk = ntk * (w_sqrt[:, None, None, None] * w_sqrt[None, None, :, None])
        ntk = ntk.reshape(2 * N_mc, 2 * N_mc)
    else:
        # Weighted double-centering. NTK is symmetric so m = K @ pdf covers both axes.
        # m[s] = sum_{s'} p_{s'} * K[s, s']
        m = ntk @ pdf                                      # [N_mc]
        m_global = jnp.dot(pdf, m)                        # scalar
        ntk = ntk - m[:, None] - m[None, :] + m_global
        # Apply W^{1/2}: tilde_K[s,s'] = sqrt(p_s)*sqrt(p_{s'}) * C[s,s']
        ntk = ntk * jnp.outer(w_sqrt, w_sqrt)
    # Note: no ntk / N_mc — the 1/N normalization is absorbed into p_s = 1/N

    # add diag shift
    # Create a sharded identity matrix to match ntk's sharding
    if config.netket_experimental_sharding:
        # Create identity matrix with same sharding as ntk: P("S", None)
        local_size = ntk.shape[0]
        identity = jnp.eye(local_size)
        identity = jax.lax.with_sharding_constraint(
            identity, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
        )
    else:
        identity = jnp.eye(ntk.shape[0])

    ntk_shifted = ntk + diag_shift * identity

    # add projection regularization
    if proj_reg is not None:
        ntk_shifted = ntk_shifted + proj_reg / N_mc

    # some solvers return a tuple, some others do not.
    if dv.ndim == 1:
        dv = jnp.expand_dims(dv, axis=1)

    aus_vector = solver_fn(ntk_shifted, dv)
    # Make sure vector is copied
    if config.netket_experimental_sharding:
        aus_vector = jax.lax.with_sharding_constraint(
            aus_vector, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
        )

    if isinstance(aus_vector, tuple):
        aus_vector, info = aus_vector
    else:
        info = {}

    if info is None:
        info = {}

    # # Center the vector, equivalent to centering
    # # The Jacobian

    # Compute weighted VJP vector: tilde_v_s = a_s*sqrt(p_s) - c*p_s
    # where c = sum_s a_s*sqrt(p_s).  For uniform p_s=1/N this reduces to
    # (a_s - mean(a)) / sqrt(N), matching the previous centering step.
    aus_vector = jnp.squeeze(aus_vector)
    if mode == "real":
        v = aus_vector * w_sqrt                               # [N_mc]
        c = jnp.sum(v, keepdims=True)                        # scalar
        aus_vector = v - c * pdf                             # [N_mc]

    if mode == "complex":
        aus_vector = aus_vector.reshape((N_mc, 2))
        v = aus_vector * w_sqrt[:, None]                      # [N_mc, 2]
        c = jnp.sum(v, axis=0, keepdims=True)                # [1, 2]
        aus_vector = v - c * pdf[:, None]                    # [N_mc, 2]

    # shape [N_mc // p.size,2]
    if config.netket_experimental_sharding:
        aus_vector = jax.lax.with_sharding_constraint(
            aus_vector,
            NamedSharding(
                jax.sharding.get_abstract_mesh(),
                P("S", *(None,) * (aus_vector.ndim - 1)),
            ),
        )

    # _, vjp_fun = jax.vjp(f, parameters_real)
    vjp_fun = nkjax.vjp_chunked(
        _apply_fn,
        parameters_real,
        samples,
        chunk_size=chunk_size,
        chunk_argnums=1,
        nondiff_argnums=1,
    )

    (updates,) = vjp_fun(aus_vector)  # pytree [N_params,]

    if momentum is not None:
        updates = tree_map(lambda x, y: x + momentum * y, updates, old_updates)
        old_updates = updates

    return rss(updates), old_updates, info
