import glob
import os
from pathlib import Path

from data_model.api.response import (
    LLMOutput, 
    MeasureSpeedOutput
)
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
            return LLMOutput(generate_text=output.text)
        except Exception as error:
            _error = str(error)
        return LLMOutput(error=_error)
    
    def measure_speed(self, prompt: str, num_iterations: int) -> MeasureSpeedOutput:
        """
        Measure speed
        :param prompt: prompt
        :param num_iterations: number of iterations
        :return:
        """
        try:
            output = self.llm_model.measure_speed(prompt, num_iterations)
            return MeasureSpeedOutput(
                tokens_per_second=output.tokens_per_second,
                avg_time_per_iteration=output.avg_time_per_iteration
            )
        except Exception as error:
            _error = str(error)
        return MeasureSpeedOutput(error=_error)