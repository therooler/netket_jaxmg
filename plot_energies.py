import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION"] = "false"

import numpy as np

from src.logger import Logger
from src.experiment_config import (ExperimentConfig,
                                   HypercubeConfig,
                                   HilbertConfig,
                                   J1J2Config,
                                   SamplerConfig,
                                   ResCNNConfig,
                                   TFIMConfig,
                                   OptimizerConfig,
                                   SRConfig)
from src.experiment_config import ConstantSchedule, CosineDecaySchedule, ExponentialDecaySchedule, pretty_config


def check_info(step, log, driver, save_path):
    if not isinstance(driver.info, dict):
        status = int(driver.info)
        if status != 0:
            print(f"Solver returned error status: {status}")
            np.save(save_path + "error_local_energies.npy", driver.state.local_estimators(driver._ham))
            print("Local energies saved...")
            return False
        else:
            return True
    return True


def get_data(ns, n_res_blocks,filters , seed):
    lattice_cfg = HypercubeConfig(L=10)
    hilbert_cfg = HilbertConfig(total_sz=0)
    critical_h = 3.044382
    tfim_cfg = TFIMConfig(J=1.,h=critical_h)
    sampler_cfg = SamplerConfig()
    rbmsymm_cfg = ResCNNConfig(n_res_blocks=n_res_blocks,
                               filters=filters,
                               kernel_shape=(3, 3), init_stddev=0.01)
    diag_shift_cfg = CosineDecaySchedule(100, decay_steps=200, end_value=1e-4)
    lr_cfg = ConstantSchedule(5e-3)
    optimizer_cfg = OptimizerConfig(lr=lr_cfg, diag_shift=diag_shift_cfg)
    sr_cfg = SRConfig(chunk_size=None)

    config = ExperimentConfig(seed=seed, n_samples=ns, n_steps=1000, thermalize_steps=10, root="./data", name="Oct21",
                              lattice=lattice_cfg, hilbert=hilbert_cfg, hamiltonian=tfim_cfg, sampler=sampler_cfg,
                              model=rbmsymm_cfg, optimizer=optimizer_cfg, sr=sr_cfg)

    save_path = config.save_path()

    fields = (("Energy", "Mean"), ("Energy", "Variance"))

    logger = Logger(path=save_path,
                    fields=fields,
                    save_every=50)
    print(save_path)
    print(os.path.exists(save_path))
    if logger.restore():
        mean_e = np.array(logger.data["Energy"]["Mean"]).real
        var_e = np.array(logger.data["Energy"]["Variance"]).real
        return mean_e, var_e, config
    else:
        return np.array([np.nan]), np.array([np.nan]), config


if __name__ == "__main__":
    ns_list = [4096, 8192, 16384, 32768, 65536,131072,196608]
    energies = []
    variances = []
    configs = []
    for ns in ns_list:
        mean_ens, var_ens, config = get_data(ns, 10,32, 100)
        energies.append(mean_ens)
        variances.append(var_ens)
        configs.append(config)

    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(8, 12)
    largest_run = 0
    n_steps = configs[0].n_steps
    for i, ns in enumerate(ns_list):
        e, v = energies[i], variances[i]
        axs[0].plot(e, label=f"{ns}")
        axs[1].plot(e, label=f"{ns}")
        axs[2].plot(v, label=f"{ns}")
    # axs[0].plot([0, n_steps], [dmrg_energy] * 2,
    #             color='gray', linestyle='--', label="Riccardos Energy")
    # axs[1].plot([0, n_steps], [dmrg_energy] * 2,
    #             color='gray', linestyle='--', label="Riccardos Energy")

    axs[0].set_ylabel("Mean Energy")
    axs[1].set_ylabel("Mean Energy")
    axs[2].set_ylabel("Variance")
    axs[2].set_yscale("log")
    axs[0].legend(loc='upper right')
    axs[0].set_ylim([-330, -100])
    axs[1].set_ylim([-327, -320])
    axs[2].set_ylim([1e0, 1e4])
    for ax in axs:
        ax.set_xlabel("Steps")
    # === Inset for diag_shift in axs[1] ===
    learning_rate = configs[0].optimizer.build_lr()
    diag_shift = configs[0].optimizer.build_diag_shift()
    axins = inset_axes(axs[1], width="25%", height="45%", loc='upper right', borderpad=3)

    axins.plot(diag_shift(np.arange(n_steps)))
    axins2 = axins.twinx()
    axins2.plot(learning_rate(np.arange(n_steps)))

    parts = configs[0].save_path().strip("/").strip("./").split("/")
    # === side text column ===
    column_text = "\n".join(parts[:-2])
    # position the text outside the axes, normalized coords
    fig.text(0.15, 0.9, column_text, va='center', ha='left', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.9', facecolor='whitesmoke', alpha=0.5))

    # adjust layout to make room on the right
    fig.subplots_adjust(top=0.8)
    axins.set_yscale('log')
    axins.set_ylabel(r'$\lambda$')
    axins.set_ylim([1e-5, 1e0])
    axins2.set_yscale('log')
    axins2.set_ylabel(r'$\eta$')
    axins2.set_ylim([1e-5, 1e0])
    axins2.tick_params(axis='both', which='major', labelsize=6)
    axins.tick_params(axis='both', which='major', labelsize=6)
    axins.grid(True, linestyle='--', alpha=0.5)
    fig.savefig(f"./figures/energies.pdf")
    plt.show()
