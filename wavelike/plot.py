import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from .pmtparam import PMTParam


def plot_waveform(
    waveform: np.ndarray | torch.Tensor,
    pmt_id: int,
    trigger_no: int,
    save_dir: str = "plots/",
    fmt: str = "pdf",
) -> None:
    """Plot the waveform for a given PMT.

    Args:
        waveform (np.ndarray | torch.Tensor): The waveform data to plot.
        pmt_id (int): The PMT identifier.
        trigger_no (int): The trigger number.
        save_dir (str): Directory to save the plot image.
        fmt (str): Format of the saved plot image.
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    os.makedirs(save_dir, exist_ok=True)
    title = f"waveform_t{trigger_no}p{pmt_id}"
    plt.figure()
    plt.plot(waveform, label=title)
    plt.title(title)
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.xlim(0, waveform.shape[0])
    plt.legend()
    plt.grid()
    plt.savefig(save_dir + title + f".{fmt}")
    plt.close()
    print(
        f"Saved waveform plot for E{trigger_no}P{pmt_id} at {save_dir + title + f'.{fmt}'}"
    )


def plot_ser(
    pmt_param: PMTParam, save_dir: str = "plots/ser_plots/", fmt: str = "pdf"
) -> None:
    """Plot the SER template for a given PMT.

    Args:
        pmt_param (PMTParam): The PMT parameters containing the SER template.
        save_dir (str): Directory to save the plot image.
        fmt (str): Format of the saved plot image.
    """
    os.makedirs(save_dir, exist_ok=True)
    title = f"ser_p{pmt_param.channel_id}"
    plt.figure()
    ser = pmt_param.ser.numpy()
    plt.plot(ser, label=title)
    plt.title(title)
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (mV)")
    plt.xlim(0, ser.shape[0])
    plt.legend()
    plt.grid()
    plt.savefig(save_dir + title + f".{fmt}")
    plt.close()
    print(
        f"Saved SER template plot for PMT{pmt_param.channel_id} at {save_dir + title + f'.{fmt}'}"
    )


def plot_prior_comparison(
    waveform: np.ndarray | torch.Tensor,
    deconv_waveform: np.ndarray | torch.Tensor,
    pe_prior: np.ndarray | torch.Tensor,
    cpe: float,
    pmt_id: int,
    trigger_no: int,
    gain: float,
    save_dir: str = "plots/prior_comparison_plots/",
    fmt: str = "pdf",
) -> None:
    """Plot comparison of original and prior results.

    Args:
        original_waveform (np.ndarray | torch.Tensor): The original waveform data.
        deconv_waveform (np.ndarray | torch.Tensor): The deconvolved waveform data.
        pe_prior (np.ndarray | torch.Tensor): The prior PE data.
        pmt_id (int): The PMT identifier.
        trigger_no (int): The trigger number.
        save_dir (str): Directory to save the plot image.
        fmt (str): Format of the saved plot image.
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    if isinstance(deconv_waveform, torch.Tensor):
        deconv_waveform = deconv_waveform.numpy()
    if isinstance(pe_prior, torch.Tensor):
        pe_prior = pe_prior.numpy()

    os.makedirs(save_dir, exist_ok=True)
    title = f"prior_comparison_t{trigger_no}p{pmt_id}"
    plt.figure()
    plt.plot(waveform, color="blue", label="Original Waveform")
    plt.plot(deconv_waveform * gain, color="green", label="Deconvolved Waveform")
    n_pe = 0
    for time, amplitude in pe_prior:
        if time != 0.0:
            plt.vlines(
                x=time, ymin=0, ymax=amplitude * gain, color="r", linestyle="--", alpha=0.5
            )
            plt.scatter(time, amplitude * gain, color="r", s=20)
            n_pe += 1
    plt.plot([], [], color="r", linestyle="--", label=f"Prior NPE: {n_pe}")
    plt.scatter([], [], color="white", s=20, label=f"CPE: {cpe:.2f}")
    plt.title(title)
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude (mV)")
    plt.xlim(100, 350)
    plt.legend()
    plt.grid()
    plt.savefig(save_dir + title + f".{fmt}")
    plt.close()
    print(
        f"Saved prior comparison plot for E{trigger_no}P{pmt_id} at {save_dir + title + f'.{fmt}'}"
    )


def plot_fit_result(
    waveform: np.ndarray | torch.Tensor,
    amps: np.ndarray,
    times: np.ndarray,
    pmt_param: PMTParam,
    pmt_id: int,
    trigger_no: int,
    gain: float,
    save_dir: str = "plots/fit_plots/",
    fmt: str = "pdf",
) -> None:
    """Plot fitted PE waveform overlaid with original waveform (baseline subtracted).

    Args:
        waveform (np.ndarray | torch.Tensor): Original waveform with baseline subtracted, shape (W,)
        amps (np.ndarray): Fitted amplitudes for each PE, shape (N,)
        times (np.ndarray): Fitted times for each PE, shape (N,)
        pmt_param (PMTParam): PMT parameters containing SER template shape
        pmt_id (int): The PMT identifier
        trigger_no (int): The trigger number
        save_dir (str): Directory to save the plot image
        fmt (str): Format of the saved plot image
    """
    from .physics import ser_waveform_numpy

    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    if isinstance(amps, torch.Tensor):
        amps = amps.numpy()
    if isinstance(times, torch.Tensor):
        times = times.numpy()

    os.makedirs(save_dir, exist_ok=True)
    title = f"fit_t{trigger_no}p{pmt_id}"

    # Reconstruct fitted waveform (same SER as nll: area = gain)
    x = np.arange(len(waveform))
    ser_matrix = ser_waveform_numpy(
        x, gain, pmt_param.decay_time, pmt_param.sigma, times
    )
    fitted_waveform = np.sum(amps * ser_matrix, axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(
        x,
        waveform,
        "b-",
        linewidth=2,
        label="Original Waveform (baseline subtracted)",
        alpha=0.7,
    )
    plt.plot(x, fitted_waveform, "r-", linewidth=2, label="Fitted Waveform", alpha=0.7)

    # Plot individual PE contributions
    for i, (amp, t) in enumerate(zip(amps, times)):
        pe_signal = amp * ser_matrix[:, i]
        plt.plot(x, pe_signal, "--", alpha=0.5, label=f"PE {i}: A={amp:.2f}, t={t:.1f}")
        # Mark PE peak position
        peak_idx = np.argmax(pe_signal)
        if peak_idx < len(x):
            plt.scatter(x[peak_idx], pe_signal[peak_idx], color="g", s=50, zorder=5)

    plt.title(f"{title} (NPE={len(amps)})")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.xlim(100, 400)
    if len(amps) <= 15:
        plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}{title}.{fmt}")
    plt.close()
    print(
        f"Saved fit result plot for E{trigger_no}P{pmt_id} at {save_dir}{title}.{fmt}"
    )
