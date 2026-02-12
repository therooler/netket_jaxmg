# NetKet + JAXMg

Variational Monte Carlo simulations for quantum many-body systems using neural quantum states with Vision Transformer and ResNet architectures. Implements distributed Stochastic Reconfiguration (SR) optimization with JAX sharding for multi-GPU scaling.

Makes use of [jaxmg](https://flatironinstitute.github.io/jaxmg/) to invert the NTK over multiple GPUs, allowing for an NTK of size $262144\times 262144$ to be inverted on 8 H200s.

If you end up using this code, please cite the corresponding references in Netket: https://netket.readthedocs.io/en/latest/cite.html
and the white paper from JAXMg:

```
@misc{2601.14466,
    Author = {Roeland Wiersema},
    Title = {JAXMg: A multi-GPU linear solver in JAX},
    Year = {2026},
    Eprint = {arXiv:2601.14466},
}
```

## Installation

```bash
pip install -r requirements.txt
```

Requires JAX 0.7.2+ with CUDA 12 support and NetKet 3.20.5+.

## Usage

Run VMC optimization:
```bash
python run_vmc.py --L 12 --ns 16384 --patch_size 2 --num_layers 2 --d_model 72 --heads 12
```

For distributed training, set up JAX environment variables:
```bash
export JAX_COORDINATOR_ADDRESS="your_coordinator:1234"
export JAX_PROCESS_COUNT=4
export JAX_PROCESS_ID=0  # Different for each process
```

Or use the provided batch scripts:
```bash
sbatch job_sub.sh
```

## Key Components

- `src/srt.py`: Optimized distributed Stochastic Reconfiguration implementation
- `src/transformer.py`: Vision Transformer architecture for quantum states
- `src/res_cnn.py`: Residual CNN architecture
- `src/vmc_sr.py`: VMC driver with SR optimization and JAXMg support.
- `run_vmc.py`: Main training script

## Visualization

Plot training results:
```bash
python plot_energies.py
python plot_memory_usage.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook plot.ipynb
```
