from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
from torch.utils.data import Dataset
import torch
import logging
import json
from copy import deepcopy
import random
from dataset.data.dataset.base_dataset import BaseDataset
from dataset.utils.io_utils import load_json
import os
from dataset.utils.io_utils import grob_paths
import torch


InputDataClass = NewType("InputDataClass", Any)


class DynamicPromptDataset(BaseDataset):
    """Dynamic prompt making dataset."""

    def __init__(self, args,
            json_data: Union[os.PathLike, List[Dict]],
            static_transform: Callable = None, 
            dynamic_transform: Callable = None, 
            shuffle: bool = False,
            from_file: bool = False,
        ):
        """
        Arguments:
            json_data (List): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            static_transform (callable): Optional transform to be applied on a sample only once.
            dynamic_transform (callable, optional): Optional transform to be applied on a sample on the fly.
        """
        if from_file:
            data = load_json(grob_paths(json_data))
        else:
            data = deepcopy(json_data)

        if shuffle:
            random.shuffle(data)
        self.data = data

        if static_transform is not None:
            self.data = list(map(lambda t: static_transform(t), self.data))

        self.dynamic_transform = dynamic_transform
        logging.info(f"data = {self.data}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, to_tensor=True):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        if self.dynamic_transform:
            sample = self.dynamic_transform(sample)

        if to_tensor:
            for k, v in sample.items():
                sample[k] = torch.tensor(v)
        # print(sample)
        return sample


class COAIDynamicPromptDataset(DynamicPromptDataset):
    def __getitem__(self, idx, to_tensor=True):
        sample = super().__getitem__(idx=idx, to_tensor=to_tensor)
        input_ids = sample["input_ids"][:-1]
        labels = sample["labels"][1:]
        attention_mask = sample["attention_mask"][1:]
        return {'input_ids': input_ids, 'attention_mask': attention_mask}, {'labels': labels}


if __name__ == '__main__':
    data_file = "/opt/tiger/llama/finetune/alpaca-lora/codes/data/alpaca_data_cleaned.json"
    train_data = DynamicPromptDataset(
        json_data=data_file, 
        dynamic_transform=lambda t: t, 
        shuffle=True, 
        from_file=True
    )

    for data in train_data:
        print(data)