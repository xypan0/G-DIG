from typing import Dict

from .base_prompt_maker import BasePromptMaker
import random


class PromptMaker(BasePromptMaker):

    def get_input(self, data_point, **kargs) -> str:
        if data_point['trg_lang']:
            trg_lang=data_point['trg_lang']
        res = f"""Translate the following text into {trg_lang}

Text:
\"{data_point["src_text"]}\"

"""
        # print(res)
        # res='test'
        return res

    def get_target(self, data_point, **kargs) -> str:
        # target = 'test'
        target = data_point["trg_text"]
        return target
    
    def get_full(self, data_point, **kargs) -> str:
        text = self.get_input(data_point) + self.get_target(data_point)
        return text