from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
import logging
import os
from dataset.utils.io_utils import grob_paths


class StreamDynamicPromptDataset(IterableDataset):
    """
    stream for large scale data
    """

    def __init__(self, args,
            json_data: Union[os.PathLike, List[Dict]],
            static_transform: Callable = None, 
            dynamic_transform: Callable = None, 
            shuffle: bool = True,
            from_file: bool = True,
        ):
        """
        json_data: json filenames splited by ",": i.e., /opt/tiger/json_data1/*,/opt/tiger/json_data2/*
        For shard/iterable dataset, static transformation is not supported.
        """
        assert from_file == True, "for StreamDynamicPromptDataset, the json data should be from file"
        logging.warning(f"static_transform is deprecated for our StreamDynamicPromptDataset.")

        json_filenames = json_data
        if from_file:
            self.data_files = grob_paths(json_filenames)
        else:
            self.data_files = json_filenames

        self.dataiter = load_dataset(
            "json", 
            data_files=self.data_files, 
            split="train", 
            streaming=True, 
            keep_in_memory=True
        )

        if shuffle:
            self.dataiter = self.dataiter.shuffle(buffer_size=vars(args).get("buffer_size", -1), seed=args.seed)
        
        if dynamic_transform:
            self.dataiter = self.dataiter.map(lambda t: dynamic_transform(t))

        logging.info(f"loading from {self.n_files} file: {self.data_files}")
    
    @property
    def n_files(self):
        return len(self.data_files)

    def __iter__(self):
        return iter(self.dataiter)


class COAIStreamDynamicPromptDataset(StreamDynamicPromptDataset):

    def __init__(self, args,
            json_data: Union[os.PathLike, List[Dict]],
            static_transform: Callable = None, 
            dynamic_transform: Callable = None, 
            shuffle: bool = True,
            from_file: bool = True,
        ):
        super().__init__(
            args=args,
            json_data=json_data,
            static_transform=static_transform,
            dynamic_transform=None,
            shuffle=shuffle,
            from_file=from_file
        )
        if dynamic_transform:
            def coai_transform(t):
                sample = dynamic_transform(t)
                input_ids = sample["input_ids"][:-1]
                labels = sample["labels"][1:]
                attention_mask = sample["attention_mask"][1:]
                return {'input_ids': input_ids, 'attention_mask': attention_mask}, {'labels': labels}
            self.dataiter = self.dataiter.map(lambda t: coai_transform(t))


if __name__ == '__main__':
    data_file = "/opt/tiger/llama/finetune/alpaca-lora/codes/data/alpaca_data_cleaned.json"
    data_files = [data_file]
    for i in range(1000):
        os.system(f"cp {data_file} /opt/tiger/json_data/{i}.json")
        data_files.append(f"/opt/tiger/json_data/{i}.json")

    shuffled_iterable_dataset = StreamDynamicPromptDataset(
        data_files, 
        shuffle=True, 
        buffer_size=100,
        dynamic_transform=lambda t: {"instruction": "1" + t["instruction"]},
    )
    for i, example in enumerate(shuffled_iterable_dataset):  # as fast as before
        print(example)
