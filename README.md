# Tensor Image Domain Gridding

Image Domain Gridding (IDG) is a fast method for convolutional resampling (gridding/degridding) of radio astronomical data (visibilities). It's highly parallel in natures making it suitable candidate for hardware accellerators like GPUs and FPGAs.

This is a GPU accelerated tensor implementation using JAX of [Image Domain Gridding](https://idg.readthedocs.io/) from ASTRON. This notebook contains the example implementation of IDG and imaging. This starts from reading the measurement set, performing the gridding, upto creating a dirty image (currently in lmn, not in RA/DEC yet).

## What is a "Tensor Implementation"

https://idg.readthedocs.io/en/latest/

## Modelling Computation as tensors

Linear algebra expressions can be represented elegantly in the tensor notation. A lot of modern numerical computation libraries offer an easy and performant ways to represent tensors and tensor operations like einsum. Modelling computation in this way keeps the code close to the domain, and makes it easy for a domain expert build new solutions.

JAX is a numerical computation library from Google that supports GPUs and TPUs. There are a wide variety of examples of JAX to create large scale ML models. JAX works equally well for creating harware accelerated simulations, and computation libraries.

Linear Algebra can be naturally represented as tensor arithmetic. For example a matrix multiplication `A x B` can be represented in the 

**Matrix multiplication**

Einstein notation

$$
C_{ij} = A_{ik} B_{kj}
$$

Code

```py
C = np.einsum("ik,kj->ij", A, B)
```

**A more complex tensor contraction**

Einstein notation

$$
D_{ij} = A_{ijk} B_{jkx} C_{ijx}
$$

Code

```
A = np.randn(2, 3, 4)
B = np.randn(3, 4, 5)
C = np.randn(2, 3, 5)

Tensor reduction
np.einsum("ijk,jkx,ijx->ij", A, B, C)
```

**Gridding**

```
phase_off = jnp.broadcast_to(jnp.einsum("px,x->p", lmn, uvw_off), (NR_TIMESTEPS, NR_CHANNELS, NR_PIXELS))
phase_coefficient = jnp.einsum("px,tx,c->tcp", lmn, uvw, wave_numbers)
phase = phase_off - phase_coefficient
phasor = lax.complex(jnp.cos(phase), jnp.sin(phase)) # Calculate phasor
pixels = jnp.einsum("tcz,tcp->zp",visibilities,phasor) # Grid with visibilty

```

## Source

Please have a look at the [notebook](./Tensor_Clean.ipynb)

## TODO

1. Flagged rows (due to RFI, instrument issues) are not removed in this example notebook during gridding. This will affect the result when using a measurement set with flagged roms
