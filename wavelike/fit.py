import math
import numpy as np
from iminuit import Minuit
from typing import Tuple
from numba import jit
from .pmtparam import PMTParam
from .utils import timer
from .physics import ser_waveform_jit


# @jit(nopython=True, fastmath=True)
# def _nll_core_dh_temp(waveform, noise_sigma, gain,
#                   charge_grid, charge_pdf,
#                   ser_gain, ser_peak_offset,
#                   amps, times):
#     """Fused DH nll: pre-computed SER template shift-add, no exp/erf at runtime."""
#     W = len(waveform)
#     N = len(amps)
#     nt = len(ser_gain)

#     # Build model via template shift-add
#     model = np.zeros(W)
#     for j in range(N):
#         tj = times[j]
#         i0_float = tj - ser_peak_offset
#         i0 = int(math.floor(i0_float))
#         frac = i0_float - i0
#         for k in range(nt - 1):
#             idx = i0 + k
#             if idx < 0 or idx >= W:
#                 continue
#             model[idx] += amps[j] * (ser_gain[k] * (1.0 - frac) + ser_gain[k + 1] * frac)

#     # Voltage NLL
#     nll = 0.0
#     for i in range(W):
#         r = waveform[i] - model[i]
#         nll += r * r
#     nll = 0.5 * nll / (noise_sigma * noise_sigma)

#     # Charge NLL
#     for j in range(N):
#         q = amps[j] * gain
#         p = np.interp(q, charge_grid, charge_pdf)
#         if p < 1e-300 or q <= 0.0:
#             p = 1e-300
#         nll -= math.log(p)

#     return nll


@jit(nopython=True, fastmath=True)
def _nll_core_dh(x, waveform, noise_sigma, gain, decay_time, sigma,
                      charge_grid, charge_pdf, amps, times):
    """Reference: inlined SER with cutoff, no template."""
    W = len(waveform)
    N = len(amps)
    nll = 0.0

    a = gain / (2.0 * decay_time)
    sig2 = sigma * sigma
    d2 = 2.0 * decay_time * decay_time
    denom = math.sqrt(2.0) * decay_time * sigma
    cutoff_left = -5.0 * sigma
    cutoff_right = 10.0 * decay_time

    for i in range(W):
        model = 0.0
        xi = x[i]
        for j in range(N):
            dt = xi - times[j]
            if dt < cutoff_left or dt > cutoff_right:
                continue
            exp_arg = sig2 / d2 - dt / decay_time
            f2 = math.exp(exp_arg)
            erf_arg = (decay_time * dt - sig2) / denom
            f3 = 1.0 + math.erf(erf_arg)
            model += amps[j] * a * f2 * f3
        r = waveform[i] - model
        nll += r * r

    nll = 0.5 * nll / (noise_sigma * noise_sigma)

    for j in range(N):
        q = amps[j] * gain
        p = np.interp(q, charge_grid, charge_pdf)
        if p < 1e-300 or q <= 0.0:
            p = 1e-300
        nll -= math.log(p)

    return nll


class WaveformFitter:
    def __init__(self, waveform: np.ndarray, pmt_param: PMTParam, noise_sigma: float = 1.0):
        self.waveform = waveform
        self.pmt_param = pmt_param
        self.noise_sigma = noise_sigma
        self.x = np.arange(len(waveform))
        if pmt_param.amplitude <= 0 or pmt_param.decay_time <= 0 or pmt_param.sigma <= 0:
            print(f"Warning: Invalid SER parameters for channel {pmt_param.channel_id}: amplitude={pmt_param.amplitude}, decay_time={pmt_param.decay_time}, sigma={pmt_param.sigma}")
            self.ser_valid = False
        else:
            self.ser_valid = True
        self.charge_model = pmt_param.charge_model

        # Pre-calculate constants for charge prior
        self.gain = pmt_param.gm # Gm

        # MultiGaussian parameters
        self.ratio = pmt_param.ratio
        self.mean = pmt_param.mean
        self.sig2 = pmt_param.sig2
        self.gauss_no = pmt_param.gauss_no
        # GammaTWeedie parameters
        # self.frac = pmt_param.frac
        # self.mu_gm = pmt_param.mu_gm
        # self.sig_gm = pmt_param.sig_gm
        # self.lam_td = pmt_param.lam_td
        # self.mu_td = pmt_param.mu_td
        # self.sig_td = pmt_param.sig_td
        # if self.charge_model == "GT":
        #     self._precompute_tweedie(self.lam_td, self.mu_td, self.sig_td)
        # DataHist: grid pre-computed in PMTParam, used via np.interp at nll time

    def _charge_pdf_multigaussian(self, charges: np.ndarray) -> np.ndarray | None:
        if self.ratio is None or self.mean is None or self.sig2 is None:
            return None
        if len(self.ratio) == 0 or len(self.mean) == 0 or len(self.sig2) == 0:
            return None
        if np.any(self.ratio <= 0) or np.any(self.mean <= 0) or np.any(self.sig2 <= 0):
            return None
    
        q = charges[:, np.newaxis]
        mu = self.mean[np.newaxis, :]
        sig = np.sqrt(self.sig2[np.newaxis, :])
        r = self.ratio[np.newaxis, :]
        gaus = (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(-0.5 * ((q - mu) / sig)**2)
        prob = np.sum(r * gaus, axis=1)
        return np.maximum(prob, 1e-300)
    
    # def _gamma(self, x, mu, sigma):
    #     alpha = mu**2 / sigma**2
    #     scale = sigma**2 / mu
    #     return gamma.pdf(x, alpha, loc=0, scale=scale)
    # '''
    # def _tweedie(self, x, lam, mu, sigma):
    #     n = 50
    #     if x == 0:
    #         return poisson.pmf(0, lam)
    #     tw = 0
    #     for i in range(n + 1):
    #         tw += poisson.pmf(i + 1, lam) * self._gamma(x, (i + 1) * mu, (1 + i) * sigma)
    #     tw += (1 - poisson.cdf(i + 1, lam)) * self._gamma(x, (i + 1) * mu, (1 + i) * sigma)
    #     return tw
    # '''
    # def _precompute_tweedie(self, lam, mu, sigma, n=30):
    #     self.tweedie_n = n
    #     self.tweedie_lam = lam
    #     self.tweedie_mu = mu
    #     self.tweedie_sigma = sigma
    #     n_vals = np.arange(1, n + 2)
    #     self.tweedie_n_vals = n_vals
    #     self.tweedie_poisson_probs = poisson.pmf(n_vals, lam)
    #     self.tweedie_alpha_base = mu * mu / (sigma * sigma)
    #     self.tweedie_scale = sigma * sigma / mu
    #     self.tweedie_log_scale = np.log(self.tweedie_scale)
    #     alpha_n = n_vals * self.tweedie_alpha_base
    #     self.tweedie_log_gamma_alpha = gammaln(alpha_n)

    # def _tweedie(self, x):
    #     x = np.asarray(x)
    #     result = np.full_like(x, np.exp(-self.tweedie_lam))
    #     mask = x > 0
    #     if not np.any(mask):
    #         return result
    #     x_nz = x[mask]
    #     log_x = np.log(x_nz)

    #     alpha_n = self.tweedie_n_vals[:, np.newaxis] * self.tweedie_alpha_base
    #     log_pdf = (alpha_n - 1) * log_x - x_nz / self.tweedie_scale - alpha_n * self.tweedie_log_scale
    #     log_pdf -= self.tweedie_log_gamma_alpha[:, np.newaxis]
    #     gamma_vals = np.exp(log_pdf)
    #     result_nz = np.sum(self.tweedie_poisson_probs[:, np.newaxis] * gamma_vals, axis=0)

    #     result[mask] = result_nz
    #     return result
    
    # def _charge_pdf_gammatweedie(self, charges: np.ndarray) -> np.ndarray:
    #     if self.frac <= 0 or self.frac >= 1:
    #         return None
    #     if self.mu_gm <= 0 or self.sig_gm <= 0:
    #         return None
    #     if self.lam_td <= 0 or self.mu_td <= 0 or self.sig_td <= 0:
    #         return None
    #     tweedie_vals = self._tweedie(charges)
    #     gamma_vals = self._gamma(charges, self.mu_gm, self.sig_gm)

    #     prob = self.frac * gamma_vals + (1 - self.frac) * tweedie_vals
    #     prob[charges <= 0] = 1e-300
    #     return np.maximum(prob, 1e-300)
        
    def nll(self, params):
        if not self.ser_valid:
            return None

        amps = params[0::2]
        times = params[1::2]

        if self.charge_model == "DH":
            return _nll_core_dh(
                self.x, self.waveform, self.noise_sigma,
                self.gain, self.pmt_param.decay_time, self.pmt_param.sigma,
                self.pmt_param.charge_grid, self.pmt_param.charge_pdf,
                amps, times,
            )
            # return _nll_core_dh_temp(
            #     self.waveform, self.noise_sigma,
            #     self.gain,
            #     self.pmt_param.charge_grid, self.pmt_param.charge_pdf,
            #     self.pmt_param.ser_gain, self.pmt_param.ser_peak_offset,
            #     amps, times,
            # )

        if self.charge_model == "MG":
            ser_matrix = ser_waveform_jit(self.x, self.gain, self.pmt_param.decay_time, self.pmt_param.sigma, times)
            model = np.sum(amps * ser_matrix, axis=1)
            nll_voltage = 0.5 * np.sum((self.waveform - model) ** 2) / (self.noise_sigma ** 2)
            charges = amps * self.gain
            prob = self._charge_pdf_multigaussian(charges)
            if prob is None:
                return None
            nll_charge = -np.sum(np.log(prob))
            return nll_voltage + nll_charge

        return None

    @timer
    def fit(self, n_pe: int, initial_guess: np.ndarray) -> Tuple[float, np.ndarray, bool]:
        """
        Perform fit for a fixed number of PEs.
        initial_guess: (n_pe, 2) array of [time, amplitude]
        Returns: (nll, parameters, valid)
        """
        if n_pe == 0:
            # 0 PE fit (baseline only)
            # model = 0
            nll = 0.5 * np.sum(self.waveform**2) / (self.noise_sigma**2)
            return nll, np.array([]), True

        # Flatten initial guess: A0, t0, A1, t1...
        start_params = []
        limits = []
        error = []
        
        for i in range(n_pe):
            t_init = initial_guess[i, 0]
            a_init = initial_guess[i, 1]
            
            # Amplitude
            start_params.append(a_init)
            limits.append((0.2, 3.)) # Amplitude > 0
            error.append(0.1)
            
            # Time
            start_params.append(t_init)
            limits.append((0, len(self.waveform))) # Time within window
            error.append(1.0)
            
        # Define NLL function for Minuit (must take unpacked arguments)
        # Use *args        
        def func_to_minimize(*args):
            result = self.nll(np.array(args))
            if result is None:  # Invalid parameters
                return float('inf')
            return result
            
        # Create Minuit
        # Generate parameter names
        par_names = []
        for i in range(n_pe):
            par_names.append(f"A{i}")
            par_names.append(f"t{i}")
            
        m = Minuit(func_to_minimize, *start_params, name=par_names)
        
        # Set limits and errors
        for i, (lim, err) in enumerate(zip(limits, error)):
            m.limits[i] = lim
            m.errors[i] = err
            
        # Run minimization
        m.print_level = 1
        m.simplex()
        m.migrad()
        
        return m.fval or float('inf'), np.array(m.values), m.valid
