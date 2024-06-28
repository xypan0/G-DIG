from typing import Dict

from .base_prompt_maker import BasePromptMaker
import random


class PromptMaker(BasePromptMaker):

    def get_input(self, data_point, **kargs) -> str:
        choise=''
        if data_point['choices']:
            choise=data_point['choices']
            choise='\n'.join([c+'.' for c in choise])
        res = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["question"]}

{choise}

### Response:
"""
        # print(res)
        # res='test'
        return res

    def get_target(self, data_point, **kargs) -> str:
        # target = 'test'
        target = data_point["solution"]
        return target
    
    def get_full(self, data_point, **kargs) -> str:
        text = self.get_input(data_point) + self.get_target(data_point)
        return text