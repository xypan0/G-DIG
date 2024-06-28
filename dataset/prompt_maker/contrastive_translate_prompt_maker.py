from typing import Dict

from .base_prompt_maker import BasePromptMaker
import random
import json

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
    
    def get_constrastive_target(self, data_point, path_contrastive_label, **kargs) -> str:
        # target = 'test'
        target = data_point["trg_text"]
        # print(path_contrastive_label)
        c_text = json.loads(open(path_contrastive_label).read())
        c_target_text=None
        for c in c_text:
            if c['src_text'] == data_point['src_text']:
                c_target_text = c['trg_text']
                break
        if c_target_text is not None:
            return c_target_text
        else:
            raise ValueError(f"Cannot match src_text: {data_point['src_text']}")
    
    def get_full(self, data_point, **kargs) -> str:
        text = self.get_input(data_point) + self.get_target(data_point)
        return text