import jax
import jax.numpy as jnp
import netket.jax as nkjax
import jax.scipy as jsp
from functools import partial
from netket.utils.types import Array
from netket.operator import AbstractOperator


@jax.jit
def ess_from_weights(w):
    s1_sq = jnp.mean(w, axis=0) ** 2
    s2 = jnp.mean(w**2, axis=0)
    # Return normalized ESS in [0, 1]
    return (s1_sq / (s2 + jnp.finfo(w.dtype).eps)).squeeze()


@jax.jit
def ess_from_weights_var(w):
    # sum over the sample axis

    s1_sq = jnp.mean(w, axis=0) ** 2
    s2 = jnp.mean(w**2, axis=0)
    # jax.debug.print("w {} s1_sq {} s2 {}",w, s1_sq, s2 )
    return ((s1_sq / (s2 - s1_sq + jnp.finfo(w.dtype).eps))).squeeze()


@partial(jax.jit, static_argnames=("apply_fn", "chunk_size"))
def blurred_sample(
    x: Array, key, params, q: float, apply_fn, op: AbstractOperator, chunk_size
):
    """One-step "bridge" proposal with importance weights.

    For each input configuration ``x[i]``, this kernel constructs a simple mixture proposal:

    - with probability ``q`` it keeps the configuration unchanged;
    - with probability ``1-q`` it proposes a *single* random connected configuration sampled
      uniformly from ``op.get_conn_padded(x[i])``.

    The returned scalar weight ``w_bridge`` corrects expectations from this mixture proposal to
    the target density :math:`p(\sigma) \propto |\psi(\sigma)|^2` (computed from
    ``apply_fn({'params': params}, ·).real``).

    Parameters
    ----------
    x:
        Array of shape ``(batch, n_dof)`` (or generally ``(batch, ...)``) containing the input
        configurations.
    key:
        JAX PRNGKey.
    params:
        Parameters passed to ``apply_fn``.
    q:
        Mixture parameter in ``[0, 1]`` controlling the probability of *staying* at the current
        configuration.
    apply_fn:
        Callable such that ``apply_fn({'params': params}, x)`` returns ``log(psi(x))`` (possibly
        complex). Only the real part is used to form :math:`|\psi|^2`.
    op:
        Operator providing ``get_conn_padded`` returning connected configurations and matrix
        elements.
    chunk_size:
        If not ``None``, evaluates the per-sample function with ``nkjax.apply_chunked``.

    Returns
    -------
    x_p:
        Array with the same shape as ``x`` containing the proposed (or unchanged) configurations.
    w_bridge:
        Array of shape ``(batch,)`` with importance weights
        :math:`w = p_{\mathrm{target}}(x_p) / p_{\mathrm{mix}}(x_p)`, where
        :math:`p_{\mathrm{target}}(\sigma) \propto |\psi(\sigma)|^2` and
        :math:`p_{\mathrm{mix}}(\sigma) = q\,p_{\mathrm{target}}(\sigma) + (1-q)\,\frac{1}{n}\sum_j p_{\mathrm{target}}(\sigma_j)`.
    E_loc:
        Local energy estimate for each proposed configuration ``x_p[i]``.
    """
    batch_size = x.shape[0]
    # rng for u1, u2 per configuration
    c = jax.random.uniform(key, shape=(batch_size, 2))

    def get_blurred_sample_and_Eloc(_in):
        _x, rng = _in
        u1, u2 = rng
        _x_shape = _x.shape
        _x = _x.reshape(-1)
        # Connected elements of Hamiltonian
        x_conn, _ = op.get_conn_padded(_x)
        # NOTE: get_conn_padded(_x) can contain diagonal elements, which correspond to "stay" configuration
        # For Ising, the first element will be diagonal, we therefore only have nconn-1 off-diagonal elements
        n_conn = x_conn.shape[-2]
        idx = jnp.floor(u2 * n_conn).astype(jnp.int32)
        # Only choose from off-diagonal elements
        proposed = x_conn[idx]
        # choose a whether to flip or stay
        x_p = jnp.where(u1 > q, _x, proposed)  # equivalent to u1 < 1-q
        x_p_conn, mels = op.get_conn_padded(x_p)
        # log |psi| for flipped and all neighbors
        logpsi_stay = apply_fn({"params": params}, x_p)
        logpsi_all = apply_fn({"params": params}, x_p_conn)
        # target density ∝ |psi|^2
        logp_stay = 2.0 * logpsi_stay.real
        logp_all = 2.0 * logpsi_all.real  # (n,)
        # stable mixture weight: (1-q)*p(stay) + (q/n)*sum_j p(all_flipped_j)
        log_term_main = jnp.log1p(-q) + logp_stay
        log_term_flips = (
            jnp.log(q) - jnp.log(n_conn) + jsp.special.logsumexp(logp_all)
        )
        log_w_bridge = jsp.special.logsumexp(jnp.stack([log_term_main, log_term_flips]))
        w_bridge = jnp.exp(logp_stay - log_w_bridge)  # scalar
        # Calculate local energies
        E_loc = jnp.sum(
            mels * jnp.exp(logpsi_all - jnp.expand_dims(logpsi_stay, -1)), axis=-1
        )
        return x_p.reshape(_x_shape), w_bridge, jnp.atleast_1d(E_loc)

    vmapped_get_blurred_sample_and_weight = jax.vmap(
        get_blurred_sample_and_Eloc, in_axes=0
    )
    if chunk_size is None:
        return vmapped_get_blurred_sample_and_weight((x, c))
    else:
        return nkjax.apply_chunked(
            vmapped_get_blurred_sample_and_weight,
            in_axes=0,
            chunk_size=chunk_size,
            axis_0_is_sharded=False,
        )((x, c))
