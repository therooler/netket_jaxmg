from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import netket.jax as nkjax
from netket.utils import timing
from netket.utils.types import Array, PyTree
from jax.sharding import NamedSharding, PartitionSpec as P
from .srt import _compute_srt_update, _compute_snr
from .mesh import create_2d_mesh, create_1d_mesh


@partial(jax.jit, static_argnames=("mode",))
def _prepare_input(
        O_L,
        local_grad,
        *,
        mode: str,
) -> tuple[jax.Array, jax.Array]:
    r"""
    Prepare the input for the SR/SRt solvers.

    The local eneriges and the jacobian are reshaped, centered and normalized by the number of Monte Carlo samples.
    The complex case is handled by concatenating the real and imaginary parts of the jacobian and the local energies.

    We use [Re_x1, Im_x1, Re_x2, Im_x2, ...] so that shards are contiguous, and jax can keep track of the sharding information.
    This format is applied both to the jacobian and to the vector.

    Args:
        O_L: The jacobian of the ansatz.
        local_grad: The local energies.
        mode: The mode of the jacobian: `'real'` or `'complex'`.

    Returns:
        The reshaped jacobian and the reshaped local energies.
    """
    N_mc = O_L.shape[0]

    local_grad = local_grad.flatten()
    de = local_grad - jnp.mean(local_grad)

    O_L = O_L / jnp.sqrt(N_mc)
    dv = 2.0 * de / jnp.sqrt(N_mc)

    if mode == "complex":
        # Concatenate the real and imaginary derivatives of the ansatz
        # (#ns, 2, np) -> (#ns*2, np)
        O_L = jax.lax.collapse(O_L, 0, 2)

        # (#ns, 2) -> (#ns*2)
        dv2 = jnp.stack([jnp.real(dv), jnp.imag(dv)], axis=-1)
        dv = jax.lax.collapse(dv2, 0, 2)
    elif mode == "real":
        dv = dv.real
    else:
        raise NotImplementedError()
    return O_L, dv


@partial(
    jax.jit,
    static_argnames=(
            "log_psi",
            "mode",
            "chunk_size",
    ),
)
def prepare_input(log_psi,
                  local_grad,
                  parameters,
                  samples,
                  model_state,
                  mode: str,
                  chunk_size: int):
    jacobians = nkjax.jacobian(
        log_psi,
        parameters,
        samples,
        model_state,
        mode=mode,
        dense=True,
        center=True,
        chunk_size=chunk_size,
    )  # jacobian is centered
    return _prepare_input(jacobians, local_grad, mode=mode)


@timing.timed
# @partial(
#     jax.jit,
#     static_argnames=(
#         "log_psi",
#         "solver_fn",
#         "mode",
#         "chunk_size",
#     ),
# )
def _sr_srt_common(
        log_psi,
        local_grad,
        parameters,
        model_state,
        samples,
        *,
        diag_shift: float | Array,
        solver_fn: Callable[[Array, Array], Array],
        mode: str,
        chunk_size: int | None = None,
):
    r"""
    Compute the SR/Natural gradient update for the model specified by
    `log_psi({parameters, model_state}, samples)` and the local gradient contributions `local_grad`.

    Uses a code equivalent to QGTJacobianDense by default, or with the NTK/MinSR if `use_ntk` is True.

    Args:
        log_psi: The log of the wavefunction.
        local_grad: The local values of the estimator.
        parameters: The parameters of the model.
        model_state: The state of the model.
        diag_shift: The diagonal shift of the stochastic reconfiguration matrix. Typical values are 1e-4 ÷ 1e-3. Can also be an optax schedule.
        proj_reg: Weight before the matrix `1/N_samples \\bm{1} \\bm{1}^T` used to regularize the linear solver in SPRING.
        momentum: Momentum used to accumulate updates in SPRING.
        linear_solver_fn: Callable to solve the linear problem associated to the updates of the parameters.
        mode: The mode used to compute the jacobian of the variational state. Can be `'real'` or `'complex'` (defaults to the dtype of the output of the model).

    Returns:
        The new parameters, the old updates, and the info dictionary.
    """
    _, unravel_params_fn = ravel_pytree(parameters)
    _params_structure = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), parameters
    )
    O_L, dv = prepare_input(log_psi, local_grad, parameters, samples, model_state, mode, chunk_size)
    mesh_2d = create_2d_mesh()
    jax.sharding.set_mesh(mesh_2d)
    updates, info = _compute_srt_update(
        O_L,
        dv,
        chunk_size=chunk_size,
        diag_shift=diag_shift,
        solver_fn=solver_fn,
        mode=mode,
        params_structure=_params_structure,
    )
    mesh_1d = create_1d_mesh()
    jax.sharding.set_mesh(mesh_1d)
    return unravel_params_fn(updates), info


@timing.timed
# @partial(
#     jax.jit,
#     static_argnames=(
#         "log_psi",
#         "mode",
#         "chunk_size",
#     ),
# )
def _srt_snr(
        log_psi,
        local_grad,
        parameters,
        model_state,
        samples,
        *,
        diag_shift: float | Array,
        mode: str,
        chunk_size: int | None = None,
):
    r"""
    Compute the SR/Natural gradient update for the model specified by
    `log_psi({parameters, model_state}, samples)` and the local gradient contributions `local_grad`.

    Uses a code equivalent to QGTJacobianDense by default, or with the NTK/MinSR if `use_ntk` is True.

    Args:
        log_psi: The log of the wavefunction.
        local_grad: The local values of the estimator.
        parameters: The parameters of the model.
        model_state: The state of the model.
        diag_shift: The diagonal shift of the stochastic reconfiguration matrix. Typical values are 1e-4 ÷ 1e-3. Can also be an optax schedule.
        proj_reg: Weight before the matrix `1/N_samples \\bm{1} \\bm{1}^T` used to regularize the linear solver in SPRING.
        momentum: Momentum used to accumulate updates in SPRING.
        linear_solver_fn: Callable to solve the linear problem associated to the updates of the parameters.
        mode: The mode used to compute the jacobian of the variational state. Can be `'real'` or `'complex'` (defaults to the dtype of the output of the model).

    Returns:
        The new parameters, the old updates, and the info dictionary.
    """
    _, unravel_params_fn = ravel_pytree(parameters)
    _params_structure = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), parameters
    )

    jacobians = nkjax.jacobian(
        log_psi,
        parameters,
        samples,
        model_state,
        mode=mode,
        dense=True,
        center=True,
        chunk_size=chunk_size,
    )  # jacobian is centered

    O_L, dv = _prepare_input(jacobians, local_grad, mode=mode)

    # TODO: Add support for proj_reg and momentum
    # At the moment SR does not support momentum, proj_reg.
    # We raise an error if they are passed with a value different from None.
    snr = _compute_snr(
        O_L,
        dv,
        diag_shift=diag_shift
    )

    return snr
