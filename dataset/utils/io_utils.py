import json
from typing import Union, List, Iterable
from glob import glob
from itertools import chain


def load_json(
        filenames: Union[str, List[str]], 
        # return_iter: bool = False
    ) -> Union[Iterable, List[str]]:
    if isinstance(filenames, str):
        return json.load(open(filenames, 'r'))
    else:
        return list(chain(*[json.load(open(filename, 'r')) for filename in filenames]))


def grob_paths(
        paths: str
    ) -> List[str]:
    if paths.startswith("\"") and paths.endswith("\""):
        paths = paths[1:-1]
    if isinstance(paths, List):
        pass
    elif isinstance(paths, str):
        paths = paths.split(",")
    else:
        raise ValueError(f"paths should be str or list of str, paths = {paths}")

    gather_paths = []
    for p in paths:
        gather_paths.extend(glob(p))
    return gather_paths


if __name__ == '__main__':
    print(grob_paths("./*"))
    print(load_json(grob_paths("../../data/*.json")))