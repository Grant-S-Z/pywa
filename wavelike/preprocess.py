import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from wavelike import PMTParam, timer
from .config import *


def get_pedestal(arr: np.ndarray, nbegin: int = bl_begin, nend: int = bl_end) -> float:
    """Get the pedestal of waveform

    Parameters
    ----------
    arr : np.ndarray
        Waveform
    nbegin : int, optional
    nend : int, optional

    Returns
    -------
    float
        Pedestal
    """    
    nbegin = max(0, nbegin)
    nend = min(window_size, nend)
    if nend <= nbegin:
        return np.mean(arr[: min(100, arr.shape[-1])]).item()
    return np.mean(arr[nbegin:nend]).item()


def get_pedestal_batch(arr_batch: torch.Tensor, nbegin: int = bl_begin, nend: int = bl_end) -> torch.Tensor:
    """Get the pedestal of waveform batch

    Parameters
    ----------
    arr_batch : torch.Tensor
        Waveform batch
    nbegin : int, optional
    nend : int, optional

    Returns
    -------
    torch.Tensor (B)
        Pedestal batch
    """    
    nbegin = max(0, nbegin)
    nend = min(window_size, nend)
    if nend <= nbegin:
        return torch.mean(arr_batch[..., : min(100, arr_batch.shape[-1])], dim=-1)
    return torch.mean(arr_batch[..., nbegin:nend], dim=-1)


def get_charge(arr: np.ndarray, ped: float, nbegin: int = inte_begin, nend: int = inte_end) -> float:
    """Get charge

    Parameters
    ----------
    arr : np.ndarray
        Waveform
    ped : float
        Baseline
    nbegin : int, optional
        Interval begin, by default inte_begin
    nend : int, optional
        Interval end, by default inte_end

    Returns
    -------
    float
        Charge
    """    
    nbegin = max(0, nbegin)
    nend = min(window_size, nend)
    if nend <= nbegin:
        return 0.0
    charge = np.sum(arr[nbegin:nend]).item() - ped * (nend - nbegin)
    return -charge


def get_charge_batch(arr_batch: torch.Tensor, ped_batch: torch.Tensor, nbegin: int = inte_begin, nend: int = inte_end) -> torch.Tensor:
    """Get charge batch

    Parameters
    ----------
    arr_batch : torch.Tensor (B, W)
        Waveform batch
    ped_batch : torch.Tensor (B)
        Baseline batch
    nbegin : int, optional
        Interval begin, by default inte_begin
    nend : int, optional
        Interval end, by default inte_end

    Returns
    -------
    torch.Tensor (B)
        Charge batch
    """    
    nbegin = max(0, nbegin)
    nend = min(window_size, nend)
    if nend <= nbegin:
        return torch.zeros(arr_batch.shape[:-1], dtype=arr_batch.dtype)
    charges = torch.sum(arr_batch[..., nbegin:nend],
                        dim=-1) - ped_batch * (nend - nbegin)
    return -charges


def get_noise(arr: np.ndarray, ped: float, nbegin: int = bl_begin, nend: int = bl_end) -> float:
    """Get noise sigma

    Parameters
    ----------
    arr : np.ndarray
        Waveform
    ped : float
        Baseline
    nbegin : int, optional
    nend : int, optional

    Returns
    -------
    float
        Noise sigma
    """
    nbegin = max(0, nbegin)
    nend = min(window_size, nend)
    if nend <= nbegin:
        nend = min(100, arr.shape[-1])
    noise_sigma = np.std(arr[..., nbegin:nend] - ped).item()
    return noise_sigma


def get_noise_batch(arr_batch: torch.Tensor, ped_batch: torch.Tensor, nbegin: int = bl_begin, nend: int = bl_end) -> torch.Tensor:
    """Get noise sigma batch

    Parameters
    ----------
    arr_batch : torch.Tensor
        Waveform batch
    ped_batch : torch.Tensor
        Baseline batch
    nbegin : int, optional
    nend : int, optional

    Returns
    -------
    torch.Tensor (B)
        Noise sigma batch
    """
    nbegin = max(0, nbegin)
    nend = min(window_size, nend)
    if nend <= nbegin:
        nend = min(100, arr_batch.shape[-1])
    noise_sigma_batch = torch.std(
        arr_batch[..., nbegin:nend] - ped_batch.unsqueeze(-1), dim=-1)
    return noise_sigma_batch


def lucyddm(waveform: np.ndarray,
            ser: np.ndarray,
            n_iter: int = 2000,
            eps: float = 1e-6) -> np.ndarray:
    """RL deconvolution

    Parameters
    ----------
    waveform : np.ndarray (W)
        Original positive sub waveform
    ser : np.ndarray (S)
        Single photoelectron response
    n_iter : int, optional
        Iteration, by default 2000
    eps : float, optional
        Minimum, by default 1e-6

    Returns
    -------
    np.ndarray (W + S - 1)
        RL deconvolution result
    """
    S = ser.shape[0]
    W = waveform.shape[0]
    L = W + S - 1  # deconvolved length

    waveform = np.clip(waveform, a_min=eps, a_max=None)
    ser = np.clip(ser, a_min=eps, a_max=None)

    deconv = np.full((L,), fill_value=eps, dtype=waveform.dtype)
    deconv[S - 1:] = waveform
    
    ser_mirror = ser[::-1]
    ser_rfft = np.fft.rfft(ser, n=L)
    ser_mirror_rfft = np.fft.rfft(ser_mirror, n=L)

    for i in range(n_iter):
        conv_result = np.fft.irfft(np.fft.rfft(
            deconv, n=L) * ser_rfft, n=L)
        relative_blur = waveform / conv_result[S - 1:]
        correction = np.fft.irfft(np.fft.rfft(
            relative_blur, n=L) * ser_mirror_rfft, n=L)
        deconv *= correction

    return deconv


def lucyddm_batch(waveform_batch: torch.Tensor,
                  ser_batch: torch.Tensor,
                  n_iter: int = 2000,
                  eps: float = 1e-6) -> torch.Tensor:
    """Batched RL deconvolution using PyTorch

    Parameters
    ----------
    waveform_batch : torch.Tensor (B, W)
        Original positive sub waveform batch
    ser_batch : torch.Tensor (B, S)
        Single photoelectron response batch
    n_iter : int, optional
        Iteration, by default 2000
    eps : float, optional
        Minimum, by default 1e-6

    Returns
    -------
    torch.Tensor (B, W + S - 1)
        RL deconvolution result
    """
    S = ser_batch.shape[1]
    B, W = waveform_batch.shape
    L = W + S - 1  # deconvolved length

    waveform_batch = torch.clamp(waveform_batch, min=eps)
    ser_batch = torch.clamp(ser_batch, min=eps)

    deconv_batch = torch.full(
        (B, L), fill_value=eps, dtype=waveform_batch.dtype, device=waveform_batch.device)
    deconv_batch[:, S - 1:] = waveform_batch
    # deconv_batch[:, C : C + W] = waveform_batch

    ser_mirror_batch = torch.flip(ser_batch, dims=[1])

    ser_rfft_batch = torch.fft.rfft(ser_batch, n=L)
    ser_mirror_rfft_batch = torch.fft.rfft(ser_mirror_batch, n=L)

    for i in range(n_iter):
        # conv_result = torch.clamp(torch.fft.irfft(torch.fft.rfft(deconv_batch, n=L) * ser_rfft_batch, n=L), min=eps)
        conv_result = torch.fft.irfft(torch.fft.rfft(
            deconv_batch, n=L) * ser_rfft_batch, n=L)
        # relative_blur = waveform_batch / conv_result[:, C : C + W]
        relative_blur = waveform_batch / conv_result[:, S - 1:]
        # correction = torch.clamp(torch.fft.irfft(torch.fft.rfft(relative_blur, n=L) * ser_mirror_rfft_batch, n=L), min=eps)
        correction = torch.fft.irfft(torch.fft.rfft(
            relative_blur, n=L) * ser_mirror_rfft_batch, n=L)
        deconv_batch *= correction

    return deconv_batch


def prior(waveform: np.ndarray,
          threshold: float = 0.15,
          min_distance: int = 5,
          max_pe: int = max_pe) -> np.ndarray:
    """Clip and cluster deconvolved waveform to give prior PE (NumPy version)

    Parameters
    ----------
    waveform : np.ndarray (W)
        Deconvolved waveform
    threshold : float, optional
        Clip threshold, by default 0.15
    min_distance : int, optional
        Minimum distance between peaks, by default 5
        (e.g., 5 means peaks must be separated by at least 5 // 2 = 2 zero samples)
    max_pe : int, optional
        Maximum number of PE, by default 200

    Returns
    -------
    np.ndarray (n_valid, 2)
        Prior PE with time and amplitude, where n_valid <= max_pe
        Only returns non-zero entries (variable length!)
    """
    # 1. Charge map: Sliding sum using convolution
    # mode='same' ensures the output size matches input and is centered
    charge_map = np.convolve(waveform, np.ones(min_distance, dtype=waveform.dtype), mode='same')

    # 2. Minimum distance suppression: Sliding max (Max Pooling)
    # Pad input to handle boundaries similar to 'same' padding
    pad_width = min_distance // 2
    padded_wave = np.pad(waveform, (pad_width, pad_width), mode='constant', constant_values=0)
    
    # Create sliding windows (requires numpy >= 1.20)
    windows = np.lib.stride_tricks.sliding_window_view(padded_wave, min_distance)
    s_map_padded = np.max(windows, axis=-1)

    # Handle potential shape mismatch if min_distance is even
    if s_map_padded.shape[0] > waveform.shape[0]:
        s_map_padded = s_map_padded[:waveform.shape[0]]

    # 3. NMS Mask: Compare waveform with local max
    # Use isclose for robust float comparison
    nms_mask = np.isclose(waveform, s_map_padded, atol=1e-8, rtol=0.0)
    mask = (waveform >= threshold) & nms_mask

    # 4. Extract valid peaks
    prior_weights = np.zeros_like(waveform, dtype=np.float32)
    prior_weights[mask] = charge_map[mask].astype(np.float32)

    k_final = min(max_pe, waveform.shape[-1])
    if k_final <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    # 5. Select Top-K
    # Use argpartition for efficiency, then sort the top k results by amplitude descending
    top_indices = np.argpartition(-prior_weights, k_final - 1)[:k_final]
    
    # Sort selected indices by amplitude (descending)
    order = np.argsort(-prior_weights[top_indices])
    top_indices = top_indices[order]
    
    top_amps = prior_weights[top_indices]
    top_times = top_indices.astype(np.float32)

    # 6. Stack results - only return valid entries
    valid_mask = top_amps > 0.0
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    pe_prior = np.zeros((n_valid, 2), dtype=np.float32)
    pe_prior[:, 0] = top_times[valid_mask]
    pe_prior[:, 1] = top_amps[valid_mask]

    return pe_prior


def prior_batch(waveform_batch: torch.Tensor,
                threshold: float = 0.15,
                min_distance: int = 5,
                max_pe: int = max_pe):
    """Clip and cluster deconvolved waveform batch to give prior PE

    Parameters
    ----------
    waveform_batch : torch.Tensor
        Deconvolved waveform batch
    threshold : float, optional
        Clip threshold, by default 0.15
    min_distance : int, optional
        Minimum distance between peaks, by default 5
        (e.g., 5 means peaks must be separated by at least 5 // 2 = 2 zero samples)
    max_pe : int, optional
        Maximum number of PE, by default 200

    Returns
    -------
    torch.Tensor (B, max_pe, 2)
        Prior PE batch with time and amplitude
    """
    # Charge map
    charge_kernel = torch.ones(1, 1, min_distance, device=waveform_batch.device)
    charge_padding = min_distance // 2

    charge_map = F.conv1d(waveform_batch.unsqueeze(1),
                          weight=charge_kernel,
                          padding=charge_padding).squeeze(1)

    # Minimum distance suppression
    kernel_size = min_distance
    padding_val = kernel_size // 2

    s_map_padded = F.max_pool1d(waveform_batch.unsqueeze(1),
                                kernel_size=kernel_size,
                                stride=1,
                                padding=padding_val).squeeze(1)

    nms_mask = (waveform_batch == s_map_padded)

    mask = (waveform_batch >= threshold) & nms_mask

    # Extract valid peaks
    prior_weights = torch.zeros_like(waveform_batch)
    prior_weights[mask] = charge_map[mask]

    k_final = min(max_pe, waveform_batch.shape[-1])
    top_amps, top_indices = torch.topk(prior_weights, k=k_final, dim=-1)
    
    # Stack
    top_times = top_indices.float()
    valid_mask = top_amps > 0.0

    pe_prior_batch = torch.stack(
        [top_times, top_amps], dim=-1)  # (B, max_hits, 2)
    pe_prior_batch *= valid_mask.unsqueeze(-1).float()

    return pe_prior_batch

@timer
def preprocess_waveform(waveform: np.ndarray,
                        ser: np.ndarray,
                        gain: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Preprocess single waveform: pedestal subtraction and deconvolution

    Parameters
    ----------
    waveform : np.ndarray (W)
        Original waveform
    ser : np.ndarray (S)
        Single photoelectron response
    gain : float
        Gain value

    Returns
    -------
    Tuple[np.ndarray (W), np.ndarray (W), np.ndarray (W), np.ndarray (max_pe, 2), float, float]
        Tuple of (pedestal-subtracted waveform, normalized waveform, deconvolved waveform, prior PE, estimated PE number by charge, noise sigma)
    """
    S = ser_length

    ped = get_pedestal(waveform)
    charge = get_charge(waveform, ped)
    noise_sigma = get_noise(waveform, ped)
    waveform_sub = ped - waveform  # sub baseline and positive

    waveform_norm = waveform_sub / gain  # waveform normalization
    cpe = charge / gain  # estimated PE number by charge

    deconv_norm = lucyddm(waveform_norm, ser)
    deconv_same = deconv_norm[S - 1: S - 1 + window_size]

    pe_prior = prior(deconv_same)

    return waveform_sub, waveform_norm, deconv_same, pe_prior, cpe, noise_sigma


def preprocess_waveform_batch(waveform_batch: torch.Tensor,
                                   ids_batch: np.ndarray,
                                   pmt_params: Dict[int, 'PMTParam'],
                                   device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess waveform batch: pedestal subtraction and deconvolution

    Parameters
    ----------
    waveform_batch : torch.Tensor (B = N * C, W)
        Original waveform batch
    ids_batch : np.ndarray (B, 2)
        Numpy array of [trigger number, PMT ID] pairs corresponding to waveform_batch
    pmt_params : Dict[int, 'PMTParam']
        PMT parameters
    device : str, optional
        Running device, by default 'cuda'

    Returns
    -------
    Tuple[torch.Tensor (B, W), torch.Tensor (B, W), torch.Tensor (B, W), torch.Tensor (B, max_pe, 2), torch.Tensor (B), torch.Tensor (B)]
        Tuple of (pedestal-subtracted waveform batch, normalized waveform batch, deconvolved waveform batch, prior PE batch, estimated PE number by charge batch, noise sigma batch)
    """
    dev = torch.device(device)
    waveform_batch = waveform_batch.to(dev).float()
    B, W = waveform_batch.shape
    S = ser_length

    pedestal_batch = get_pedestal_batch(waveform_batch)
    charge_batch = get_charge_batch(waveform_batch, pedestal_batch)
    noise_batch = get_noise_batch(waveform_batch, pedestal_batch)
    waveform_batch = pedestal_batch.unsqueeze(-1) - waveform_batch  # sub baseline and positive

    ser_batch = torch.zeros((B, S), dtype=torch.float, device=dev)
    gain_batch = torch.ones((B,), dtype=torch.float, device=dev)

    for i, (_, pid) in enumerate(ids_batch):
        pmt_param = pmt_params.get(pid)
        if pmt_param is None:
            print(
                f"Warning: PMT ID {pid} parameters not found. Using default values.")
            continue
        else:
            ser = pmt_param.ser.to(dev)
            ser /= torch.sum(ser)  # ser normalization
            ser_batch[i, :] = ser
            gain_batch[i] = pmt_param.gain

    waveform_norm_batch = waveform_batch / \
        gain_batch.unsqueeze(-1)  # waveform normalization
    
    cpe_batch = charge_batch / gain_batch  # estimated PE number by charge

    deconv_norm_batch = lucyddm_batch(waveform_norm_batch, ser_batch)
    deconv_same = deconv_norm_batch[:, S - 1: S - 1 + W].contiguous()

    pe_prior_batch = prior_batch(deconv_same)

    return waveform_batch, waveform_norm_batch, deconv_same, pe_prior_batch, cpe_batch, noise_batch

