from typing import Dict

class BasePromptMaker():
    def __init__(self, *args, **kargs):
        pass

    def get_input(self, data_point: Dict[str, str], **kargs) -> str:
        raise NotImplementedError()

    def get_target(self, data_point: Dict[str, str], **kargs) -> str:
        raise NotImplementedError()
    
    def get_full(self, data_point: Dict[str, str], **kargs) -> str:
        raise NotImplementedError()
