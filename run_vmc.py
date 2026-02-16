import os
import platform
import time

from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_analytical_sol_latency_estimator=false"
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
        time.sleep(1)
        import jax

        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=world_size,
            process_id=rank,
            local_device_ids=rank,
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

jax.config.update("jax_enable_x64", True)
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
    J1J2Config,
    HeisenbergConfig,
    TriangularConfig,
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

import platform
from functools import partial
from jax.sharding import PartitionSpec as P, NamedSharding

system = platform.system()

if system == "Linux":
    from jaxmg import potrs

    JAXMG_ENABLED = True
else:
    JAXMG_ENABLED = False


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
    return ~jnp.isnan(driver._loss_stats.mean)


fields_to_track = (("Energy", "Mean"), ("Energy", "Variance"))


def main(args, return_logger=False):
    lattice_cfg = TriangularConfig(L=args.L, max_neighbor_order=1)
    hilbert_cfg = HilbertConfig(total_sz=0)
    # critical_h = 3.044382
    # tfim_cfg = TFIMConfig(J=1.0, h=critical_h)
    # J2=0.0
    # system_cfg = J1J2Config(J=(1.0, J2))
    system_cfg = HeisenbergConfig(sign_rule=None)
    sampler_cfg = SamplerConfig()
    model_cfg = ViT2DConfig(
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        d_model=args.d_model,
        heads=args.heads,
    )
    diag_shift_cfg = ConstantSchedule(args.diag_shift)
    # ConstantSchedule(value=0.01)
    lr_init = args.lr
    lr_cfg = CosineDecaySchedule(lr_init, 10000, lr_init / 10)
    optimizer_cfg = OptimizerConfig(lr=lr_cfg, diag_shift=diag_shift_cfg)
    # sr_cfg = SRConfig(chunk_size=None)
    sr_cfg = SRConfig(chunk_size=args.chunk_size)

    config = ExperimentConfig(
        seed=args.seed,
        n_samples=args.ns,
        n_steps=10000,
        thermalize_steps=args.thermalizing_steps,
        root="./data",
        name=args.experiment_name,
        lattice=lattice_cfg,
        hilbert=hilbert_cfg,
        hamiltonian=system_cfg,
        sampler=sampler_cfg,
        model=model_cfg,
        optimizer=optimizer_cfg,
        sr=sr_cfg,
    )
    save_path = config.save_path()
    if jax.process_index() == 0:
        print(f"Making path in process {jax.process_index()}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        config.save_config_yaml()
    logger = Logger(
        path=save_path, fields=fields_to_track, save_every=10, rank=jax.process_index()
    )
    if return_logger:
        return logger

    print(config)
    print(f"Save path {config.save_path()}")

    hamiltonian = config.build_hamiltonian()
    sampler = config.build_sampler()
    model = config.build_model()
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key, 2)

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        sampler_seed=subkey,
        n_samples_per_rank=sampler.n_chains_per_rank,
        chunk_size=config.sr.chunk_size,
        n_discard_per_chain=0,
        seed=subkey
    )
    npar = vstate.n_parameters
    print(f"Number of parameters {npar}")
    print(f"Memory allocated: {args.ns**2 * jnp.dtype(jnp.float64).itemsize/1e9} GB")

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
            x = vstate.sample()
        print("Thermalizing done")
        print("SR optimization")

        optimizer = optax.sgd(learning_rate=config.optimizer.build_lr())

        print(sampler.n_chains_per_rank)
        if JAXMG_ENABLED:
            print("Using JAXMg...")
            linear_solver = partial(
                potrs,
                T_A=2**12,
                mesh=jax.sharding.get_abstract_mesh(),
                in_specs=(P("S", None), P(None, None)),
            )
        else:
            linear_solver = nk.optimizer.solver.cholesky
        driver = VMC_SR(
            hamiltonian,
            variational_state=vstate,
            optimizer=optimizer,
            diag_shift=config.optimizer.build_diag_shift(),
            chunk_size_bwd=config.sr.chunk_size,
            momentum=False,
            linear_solver=linear_solver,
            q=args.q,
        )
        # else:
        #     driver = VMC_SR(
        #         hamiltonian,
        #         variational_state=vstate,
        #         optimizer=optimizer,
        #         diag_shift=config.optimizer.build_diag_shift(),
        #         chunk_size_bwd=config.sr.chunk_size,
        #         momentum=False
        #     )
        # driver = nk.driver.VMC_SR(
        #     hamiltonian,
        #     variational_state=vstate,
        #     optimizer=optimizer,
        #     diag_shift=config.optimizer.build_diag_shift(),
        #     chunk_size_bwd=config.sr.chunk_size,
        #     momentum=False
        # )
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
    parser.add_argument("--ns", type=int, default=2**12, help="Number of samples")
    parser.add_argument(
        "--patch_size", type=int, default=3, help="Effective patch size"
    )
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument(
        "--d_model", type=int, default=72, help="Model hidden vector size"
    )
    parser.add_argument("--heads", type=int, default=12, help="Number of MHA heads")
    parser.add_argument("--L", type=int, default=4, help="Lattice size of 2D lattice")
    parser.add_argument(
        "--chunk_size", type=int, default=None, help="Jacobian chunk size NTK"
    )
    parser.add_argument("--seed", type=int, default=100, help="Seed")
    parser.add_argument(
        "--experiment_name", type=str, default="Feb15", help="Experiment name"
    )
    parser.add_argument(
        "--diag_shift", type=float, default=1e-3, help="Diagonal shift for SR"
    )
    parser.add_argument("--q", type=float, default=None, help="Bridge weight")
    parser.add_argument(
        "--thermalizing_steps",
        type=int,
        default=1000,
        help="Number of thermalizing steps",
    )
    parser.add_argument("--lr", type=float, default=0.0075, help="Learning rate")
    args = parser.parse_args()

    main(args)
