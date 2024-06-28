from .base_prompt_maker import BasePromptMaker
from typing import Dict, List
import random

class TranslationPromptMaker(BasePromptMaker):
    def __init__(self, data_point):
        self.data_point = data_point
    
    def get_input(self) -> str:
        prompts = self.data_point['translation']
        if len(prompts) > 1:
            prompt = random.choice(prompts)
        elif len(prompts) == 1:
            prompt = prompts[0]
        else:
            raise Exception
        return prompt

