from tqdm.auto import tqdm as original_tqdm
from functools import partial


def make_tqdm(accelerator, list_data):
    tqdm = partial(original_tqdm, disable=not accelerator.is_local_main_process, position=0)
    return tqdm(list_data)