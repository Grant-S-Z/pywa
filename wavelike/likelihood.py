import torch
import torch.nn as nn
from typing import Dict

class wavelikelihood_batch(nn.Module):
    def __init__(self,                 
                 waveform_batch: torch.Tensor, # core
                 initial_pe: torch.Tensor,
                 ser_batch: torch.Tensor,
                 cpe_batch: torch.Tensor,
                 noise_batch: torch.Tensor, # time loss
                 charge_params: Dict[str, torch.Tensor], # charge loss
                 fit_config: Dict):
        """Init wavelikelihood

        Parameters
        ----------
        waveform_batch : torch.Tensor (B, W)
            Waveform batch (sub baseline and positive)
        initial_pe : torch.Tensor (B, max_pe, 2) [time, amplitude]
            Initial PE parameters batch
        ser_batch : torch.Tensor (B, ser_length)
            Single electron response batch
        cpe_batch : torch.Tensor (B,)
            Estimated PE batch by charge
        noise_batch : torch.Tensor (B,)
            Baseline noise batch
        charge_params : Dict[str, torch.Tensor]
            Charge fitting parameters
            - 'charge_mean': torch.Tensor (B,)
            - 'charge_sigma': torch.Tensor (B,)
        fit_config : Dict
            Fitting configuration
            - 'fit_begin'
            - 'fit_end'
            - 'time_weight'
            - 'charge_weight'
            - 'device'
        """            
        super(wavelikelihood_batch, self).__init__()
        self.waveform_batch = waveform_batch
        self.initial_pe = initial_pe
        self.fit_begin = fit_config['fit_begin']
