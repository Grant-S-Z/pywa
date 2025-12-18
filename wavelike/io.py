import numpy as np
import uproot as ur
import awkward as ak
import torch
from typing import Dict, Iterator, List, Set, Tuple, Optional, cast
from .config import *

class DataReader:
    """
    Use uproot to read waveform data efficiently from ROOT files.
    """
    def __init__(self, filepath: str, tree_name: str = "Readout", allowed_pmts: Optional[Set[int]] = None):
        self.filepath = filepath
        self.tree_name = tree_name
        self.allowed_pmts = allowed_pmts
        print(f"DataReader initialized for file: {self.filepath}")

    def get_event_generator(self, batch_size: int = 100, max_batches=None) -> Iterator[Tuple[int, Dict[int, np.ndarray]]]:
        """Yield one event

        Parameters
        ----------
        batch_size : int, optional
            Number of events to read per batch, by default 100
        max_batches : int, optional
            Maximum number of batches to read, by default None

        Yields
        ------
        Iterator[Tuple[int, Dict[int, np.ndarray]]]
            TriggerNo and corresponding waveform
        """
        for batch_count, batch in enumerate(ur.iterate(f'{self.filepath}:{self.tree_name}',
                                ["TriggerNo", "ChannelId", "Waveform"],
                                library="ak",
                                step_size=batch_size # pyright: ignore[reportArgumentType]
                                )):
            batch = cast(ak.Array, batch)
            for i in range(len(batch)):
                event_trigger_no = batch.TriggerNo[i]
                event_channel_ids = batch.ChannelId[i]
                event_waveform = batch.Waveform[i]
                event_dict: Dict[int, np.ndarray] = {}

                for j, ch_id in enumerate(event_channel_ids):
                    if self.allowed_pmts is not None and int(ch_id) not in self.allowed_pmts:
                        continue
                    start = j * window_size
                    end = (j + 1) * window_size
                    
                    waveform_np = event_waveform[start:end].to_numpy()

                    event_dict[int(ch_id)] = waveform_np

                yield event_trigger_no,event_dict       

            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break 

        
    def get_batch_generator(self, batch_size: int = 100, max_batches=None) -> Iterator[Tuple[np.ndarray, torch.Tensor]]:
        """Yield flattened batch

        Parameters
        ----------
        batch_size : int, optional
            Number of events to read per batch, by default 100
        max_batches : int, optional
            Maximum number of batches to read, by default None

        Yields
        ------
        Iterator[Tuple[np.ndarray, torch.Tensor]]
            - ids_batch (B, 2) [(trigger_no, channel_id)]
            - waveform_batch (B, W)
        """ 

        for batch_count, batch in enumerate(ur.iterate(f'{self.filepath}:{self.tree_name}',
                                ["TriggerNo", "ChannelId", "Waveform"],
                                library="ak",
                                step_size=batch_size # pyright: ignore[reportArgumentType]
                                )):
            batch = cast(ak.Array, batch)            
            ids_batch_list : List[np.ndarray] = [] # (trigger_no, channel_id)
            waveform_batch_list: List[np.ndarray] = []

            for i in range(len(batch)):
                event_trigger_no = batch.TriggerNo[i]
                event_channel_ids = batch.ChannelId[i]
                event_flat_waveform = batch.Waveform[i]
                
                for j, ch_id in enumerate(event_channel_ids):
                    if self.allowed_pmts is not None and int(ch_id) not in self.allowed_pmts:
                        continue
                    start = j * window_size
                    end = (j + 1) * window_size
                    
                    waveform_np = event_flat_waveform[start:end].to_numpy()
                    
                    ids_batch_list.append(np.array([event_trigger_no, int(ch_id)]))
                    waveform_batch_list.append(waveform_np)

            ids_batch = np.stack(ids_batch_list, axis=0)  # (B, 2)
            waveform_batch_np = np.stack(waveform_batch_list, axis=0)
            waveform_batch_tensor = torch.from_numpy(waveform_batch_np).float()

            yield ids_batch, waveform_batch_tensor

            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break

class DataWriter:
    """
    Use uproot to write waveform data efficiently to ROOT files.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = ur.recreate(filepath)
        self.events_data = {
            "TriggerNo": [],
            "ChannelId": [],
            "Amplitudes": [],
            "Times": []
        }
        print(f"DataWriter: Initialized for {filepath}")

    def write_event(self, trigger_no: int, channel_id: int, amps: np.ndarray, times: np.ndarray):
        """Write one event

        Parameters
        ----------
        trigger_no : int
            Trigger number for the event
        channel_id : int
            Channel ID for the event
        amps : np.ndarray
            Amplitudes array for the event
        times : np.ndarray
            Times array for the event
        """
        self.events_data["TriggerNo"].append(trigger_no)
        self.events_data["ChannelId"].append(channel_id)
        self.events_data["Amplitudes"].append(amps)
        self.events_data["Times"].append(times)

    def close(self):
        """Finalize and write data to ROOT file"""
        data_branch = {
            "TriggerNo": "int32",
            "ChannelId": "int32",
            "Amplitudes": "var * float32",
            "Times": "var * float32"
        }

        tree = self.file.mktree("FittedWaveforms", data_branch)

        data_dict = {
            "TriggerNo": np.array(self.events_data["TriggerNo"], dtype=np.int32),
            "ChannelId": np.array(self.events_data["ChannelId"], dtype=np.int32),
            "Amplitudes": ak.Array(self.events_data["Amplitudes"]),
            "Times": ak.Array(self.events_data["Times"])
        }

        tree.extend(data_dict)
        self.file.close()
        print(f"DataWriter: Data written and closed to {self.filepath}")