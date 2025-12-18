import numpy as np
from iminuit import Minuit
from typing import Tuple
from .pmtparam import PMTParam
from .utils import timer
from .physics import ser_waveform_jit


class WaveformFitter:
    def __init__(self, waveform: np.ndarray, pmt_param: PMTParam, noise_sigma: float = 1.0):
        self.waveform = waveform
        self.pmt_param = pmt_param
        self.noise_sigma = noise_sigma
        self.x = np.arange(len(waveform))
        
        # Pre-calculate constants for charge prior
        self.gain = pmt_param.gain
        self.ratio = pmt_param.ratio
        self.mean = pmt_param.mean
        self.sig2 = pmt_param.sig2
        self.gauss_no = pmt_param.gauss_no
        
        pass

    def nll(self, params):
        # params: A0, t0, A1, t1, ...
        amps = params[0::2]
        times = params[1::2]
        
        # 1. Reconstruct Waveform
        # model = sum(A_j * SER(t - t_j))
        # ser_matrix: (W, N)
        # ser_matrix = ser_waveform_numpy(self.x, self.pmt_param.amplitude, self.pmt_param.decay_time, self.pmt_param.sigma, times)
        ser_matrix = ser_waveform_jit(self.x, self.pmt_param.amplitude, self.pmt_param.decay_time, self.pmt_param.sigma, times)
        
        # Multiply by amplitudes
        # amps: (N,)
        # model: (W,)
        model = np.sum(amps * ser_matrix, axis=1)
        
        # 2. Voltage Likelihood (Chi2)
        # L_A = -0.5 * sum((V - f)**2) / sigma^2
        # NLL_A = 0.5 * sum((V - f)**2) / sigma^2
        residuals = self.waveform - model
        nll_voltage = 0.5 * np.sum(residuals**2) / (self.noise_sigma**2)
        
        # 3. Charge Likelihood
        # L_C = sum_j log( sum_k r_k * Gauss(A_j * Gain, mu_k, sig_k) )
        charges = amps * self.gain
        
        # Mixture of Gaussians
        q = charges[:, np.newaxis]
        mu = self.mean[np.newaxis, :]
        sig = np.sqrt(self.sig2[np.newaxis, :]) # sig2 is variance
        r = self.ratio[np.newaxis, :]
        
        # Gaussian term
        # 1/(sqrt(2pi)*sig) * exp(-0.5 * ((q - mu)/sig)**2)
        gaus = (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(-0.5 * ((q - mu) / sig)**2)
        
        # Sum over k
        prob = np.sum(r * gaus, axis=1)
        
        # Avoid log(0)
        prob = np.maximum(prob, 1e-300)
        
        nll_charge = -np.sum(np.log(prob))
        
        return nll_voltage + nll_charge

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
            limits.append((0, None)) # Amplitude > 0
            error.append(0.1)
            
            # Time
            start_params.append(t_init)
            limits.append((0, len(self.waveform))) # Time within window
            error.append(1.0)
            
        # Define NLL function for Minuit (must take unpacked arguments)
        # Use *args        
        def func_to_minimize(*args):
            return self.nll(np.array(args))
            
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
        # m.print_level = 1
        m.migrad()
        
        return m.fval or float('inf'), np.array(m.values), m.valid
