
import logging
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, *args, **kargs):
        logging.info(f"initiate dataset: {BaseDataset.__name__}")
    


if __name__ == '__main__':
    BaseDataset()
