import flax.linen as nn
from flax import nnx
from flax.linen import Module
from flax.nnx.nn import initializers
from netket.utils.types import NNInitFunc
import jax
import typing as tp
import jax.numpy as jnp
from jax import lax
from flax.typing import Array, Dtype, Axes
from typing import Any
from flax.linen.normalization import LayerNorm, _normalize, _canonicalize_axes, _abs_sq
from flax.linen import module

compact = module.compact
default_kernel_init = initializers.he_normal()
default_bias_init = initializers.he_normal()

# If the input is an integer gelu will upcast to f32 and return in f32
default_activation = nnx.gelu

def _reshape(x, linear_size):
    if len(x.shape) > 1:
        x = x.reshape((x.shape[:-1] + (linear_size, linear_size, 1))) # (Batch_size, L, L, in_filters=1)
    else:
        x = x.reshape((linear_size, linear_size, 1)) # (L, L, in_filters=1)
    
    return x


def _custom_compute_stats(
        x: Array,
        axes: Axes,
        dtype: Dtype | None,
        axis_name: str | None = None,
        axis_index_groups: Any = None,
        use_mean: bool = True,
        use_fast_variance: bool = True,
        mask: Array | None = None,
        upcast_sums: bool = True
):
    if dtype is None:
        dtype = jnp.result_type(x)

    x = jnp.asarray(x, dtype)
    axes = _canonicalize_axes(x.ndim, axes)

    def maybe_distributed_mean(*xs, mask=None):
        mus = tuple(x.mean(axes, where=mask, dtype=None if upcast_sums else dtype) for x in xs)
        if axis_name is None:
            return mus if len(xs) > 1 else mus[0]
        else:
            # In the distributed case we stack multiple arrays to speed comms.
            if len(xs) > 1:
                reduced_mus = lax.pmean(
                    jnp.stack(mus, axis=0),
                    axis_name,
                    axis_index_groups=axis_index_groups,
                )
                return tuple(reduced_mus[i] for i in range(len(xs)))
            else:
                return lax.pmean(mus[0], axis_name, axis_index_groups=axis_index_groups)

    if use_mean:
        if use_fast_variance:
            mu, mu2 = maybe_distributed_mean(x, _abs_sq(x), mask=mask)
            # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
            # to floating point round-off errors.
            var = jnp.maximum(0.0, mu2 - _abs_sq(mu))
        else:
            mu = maybe_distributed_mean(x, mask=mask)
            var = maybe_distributed_mean(
                _abs_sq(x - jnp.expand_dims(mu, axes)), mask=mask
            )
    else:
        var = maybe_distributed_mean(_abs_sq(x), mask=mask)
        mu = jnp.zeros_like(var)
    return mu, var


class CustomLayerNorm(LayerNorm):
    upcast_sums: bool = True

    @compact
    def __call__(self, x, *, mask: jax.Array | None = None):
        """Applies layer normalization on the input.

        Args:
        x: the inputs
        mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
            the positions for which the mean and variance should be computed.

        Returns:
        Normalized inputs (the same shape as inputs).
        """

        mean, var = _custom_compute_stats(
            x,
            self.reduction_axes,
            self.dtype,
            self.axis_name,
            self.axis_index_groups,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
            upcast_sums=self.upcast_sums
        )

        return _normalize(
            self,
            x,
            mean,
            var,
            self.reduction_axes,
            self.feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
            self.force_float32_reductions,
        )

class Conv2D(nn.Module):
    linear_size: int
    filters: int = 1
    kernel_shape: tuple[int, int] = (1, 1)
    precision: tp.Any = None
    use_bias: bool = False
    param_dtype: any = jnp.float64
    kernel_init: NNInitFunc = initializers.lecun_normal()
    reshape: bool = True

    @nn.compact
    def __call__(self, x):
        if self.reshape:
            x = x.reshape((x.shape[:-1] + (self.linear_size, self.linear_size, 1)))  # (Batch_size, L, L, in_filters=1)

        x = nn.Conv(
            features=self.filters,
            kernel_size=self.kernel_shape,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init
        )(x)

        return x


class ResBlock(Module):
    linear_size: int
    filters: int = 1
    kernel_shape: tuple[int, int] = (1, 1)
    precision: tp.Any = None
    use_bias: bool = False
    param_dtype: any = jnp.float64
    kernel_init: NNInitFunc = default_kernel_init
    activation: tp.Any = default_activation
    upcast_sums: bool = True
    reshape: bool = True

    @nn.compact
    def __call__(self, input):
        if self.reshape: 
            input = _reshape(input, self.linear_size)

        x = CustomLayerNorm(
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            upcast_sums=self.upcast_sums 
        )(input)

        x = self.activation(x)

        x = nn.Conv(
            features=self.filters,
            kernel_size=self.kernel_shape,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init
        )(x)

        x = self.activation(x)

        x = nn.Conv(
            features=self.filters,
            kernel_size=self.kernel_shape,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init
        )(x)

        return x + input
    
class ResCNN(Module):
    linear_size: int
    n_res_blocks: int = 1
    filters: int = 1
    kernel_shape: tuple[int, int] = (1, 1)
    precision: tp.Any = None
    use_bias: bool = False
    param_dtype: any = jnp.float64
    kernel_init: NNInitFunc = default_kernel_init
    activation: tp.Any = default_activation
    upcast_sums: bool = True
    reshape: bool = True

    @nn.compact
    def __call__(self, input):
        if self.reshape: 
            input = _reshape(input, self.linear_size)
            
        x = nn.Conv(
            features=self.filters,
            kernel_size=self.kernel_shape,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init
        )(input)

        for _ in range(self.n_res_blocks):
            x = ResBlock(
                linear_size=self.linear_size,
                filters=self.filters,
                kernel_shape=self.kernel_shape,
                precision=self.precision,
                use_bias=self.use_bias,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                activation=self.activation,
                upcast_sums=self.upcast_sums,
                reshape=False
            )(x)
        
        x = CustomLayerNorm(
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            upcast_sums=self.upcast_sums 
        )(x)

        return jnp.sum(x, axis=[-1, -2, -3], dtype=None if self.upcast_sums else self.param_dtype)

    @property
    def label(self):
        return f"ResCNN_nblocks{self.n_res_blocks}_nfilters{self.filters}_KernelShape{self.kernel_shape[0]}x{self.kernel_shape[1]}"