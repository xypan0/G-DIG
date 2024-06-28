from typing import Dict

from .base_prompt_maker import BasePromptMaker
import random


class PromptMaker(BasePromptMaker):

    def get_input(self, data_point, **kargs) -> str:
        return f'''
{data_point['system_prompt']}

{data_point['question']}

'''

    def get_target(self, data_point, **kargs) -> str:
        target = data_point["response"]
        return target
    
    def get_full(self, data_point, **kargs) -> str:
        text = self.get_input(data_point) + self.get_target(data_point)
        return text