import os
import platform
import time

from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION"] = "false"

if platform.system() == "Linux":
    try:
        coordinator_address = os.environ["JAX_COORDINATOR_ADDRESS"]
        world_size = int(os.environ["JAX_PROCESS_COUNT"])
        rank = int(os.environ["JAX_PROCESS_ID"])
        print(f"coordinator_address: {coordinator_address}")
        print(f"world_size: {world_size}")
        print(f"rank: {rank}")
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        local_device_ids = [int(c) for c in cuda_visible_devices.split(",")]
        print("local_device_ids:", local_device_ids)
        time.sleep(1)
        import jax

        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=world_size,
            process_id=rank,
            local_device_ids=local_device_ids,
        )
        print("Done initializing jax")
    except KeyError:
        import jax

        print("Single node computation")
    print(jax.devices())
else:
    print("Running on MacOs")
    ndev = 4
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ndev}"
    import jax

import netket as nk

from functools import partial

import optax
from src.vmc_sr import VMC_SR

from src.logger import Logger
from src.experiment_config import (
    ExperimentConfig,
    HypercubeConfig,
    HilbertConfig,
    TFIMConfig,
    SamplerConfig,
    ResCNNConfig,
    ViT2DConfig,
    OptimizerConfig,
    SRConfig,
)
from src.experiment_config import (
    ConstantSchedule,
    CosineDecaySchedule,
    ExponentialDecaySchedule,
    pretty_config,
)
import jax.numpy as jnp


def check_info(step, log, driver, save_path):
    if not isinstance(driver.info, dict):
        status = int(driver.info)
        if status != 0:
            print(f"Solver returned error status: {status}")
            # print("Local energies saved...")
            return False
        else:
            # np.save(save_path + f"error_local_energies_{step%10}.npy", driver.state.local_estimators(driver._ham))
            return True

    return True


def main(args):
    lattice_cfg = HypercubeConfig(L=args.L)
    hilbert_cfg = HilbertConfig()
    critical_h = 3.044382
    tfim_cfg = TFIMConfig(J=1.0, h=critical_h)
    sampler_cfg = SamplerConfig()
    rbmsymm_cfg = ViT2DConfig(
        L_eff=args.L_eff,
        num_layers=args.num_layers,
        d_model=args.d_model,
        heads=args.heads,
        b=args.b
    )
    diag_shift_cfg = ConstantSchedule(1e-4)
    # ConstantSchedule(value=0.01)
    lr_cfg = ConstantSchedule(5e-3)
    optimizer_cfg = OptimizerConfig(lr=lr_cfg, diag_shift=diag_shift_cfg)
    sr_cfg = SRConfig(chunk_size=args.chunk_size)

    config = ExperimentConfig(
        seed=args.seed,
        n_samples=args.ns,
        n_steps=1000,
        thermalize_steps=1,
        root="./data",
        name="Oct21",
        lattice=lattice_cfg,
        hilbert=hilbert_cfg,
        hamiltonian=tfim_cfg,
        sampler=sampler_cfg,
        model=rbmsymm_cfg,
        optimizer=optimizer_cfg,
        sr=sr_cfg,
    )
    print(config)
    print(f"Save path {config.save_path()}")
    config.save_config_yaml()

    hamiltonian = config.build_hamiltonian()
    sampler = config.build_sampler()
    model = config.build_model()
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key, 2)

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        sampler_seed=subkey,
        n_samples_per_rank=sampler.n_chains_per_rank
    )
    npar = vstate.n_parameters
    print(f"Number of parameters {npar}")
    print(f"Memory allocated: {args.ns**2 * jnp.dtype(jnp.float64).itemsize/1e9} GB")
    save_path = config.save_path()
    fields = (("Energy", "Mean"), ("Energy", "Variance"))
    if jax.process_index() == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    logger = Logger(
        path=save_path, fields=fields, save_every=10, rank=jax.process_index()
    )
    restored = logger.restore(vstate)
    if restored:
        step = logger.data["iters"]["values"][-1]
        # logger.restore_samples(vstate)
        print("Restored step:", step)
        print("Last energy:", logger.data["Energy"]["Mean"][-1])

    else:
        step = 0
    if step < config.n_steps - 1:
        print("Thermalizing...")
        for _ in tqdm(range(config.thermalize_steps)):
            vstate.sample()
        print("Thermalizing done")
        print("SR optimization")

        optimizer = optax.sgd(learning_rate=config.optimizer.build_lr())
        driver = VMC_SR(
            hamiltonian,
            variational_state=vstate,
            optimizer=optimizer,
            diag_shift=config.optimizer.build_diag_shift(),
            chunk_size_bwd=config.sr.chunk_size,
        )
        driver._step_count = step
        driver.run(
            config.n_steps - step,
            out=logger,
            callback=partial(check_info, save_path=save_path),
        )

    print("Done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parser for training RNN WF")
    # Optional argument
    parser.add_argument("--ns", type=int, default=2**16, help="Number of samples")
    parser.add_argument("--L_eff", type=int, default=25, help="Effective patch size")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--d_model", type=int, default=36, help="Model hidden vector size" )
    parser.add_argument("--heads", type=int, default=4, help="Number of MHA heads")
    parser.add_argument("--b", type=int, default=2, help="b patch")
    parser.add_argument("--L", type=int, default=10, help="Lattice size of 2D lattice")
    parser.add_argument( "--chunk_size", type=int, default=2**12, help="Jacobian chunk size NTK")
    parser.add_argument("--seed", type=int, default=100, help="Seed")
    args = parser.parse_args()

    main(args)
