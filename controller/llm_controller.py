import glob
import os
from pathlib import Path

from data_model.api.response import LLMOutput
from utils.base import Base


class LLMController(Base):
    def __init__(
            self,
            llm_model,
    ):
        super(LLMController, self).__init__()
        self.llm_model = llm_model

    def generate_text(self, prompt: str, max_length: int) -> LLMOutput:
        """
        Generate text
        :param prompt: prompt
        :return:
        """
        try:
            output = self.llm_model.generate_text(prompt, max_length)
            return LLMOutput(generate_text=output)
        except Exception as error:
            _error = str(error)
        return LLMOutput(error=_error)