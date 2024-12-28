from typing import Any, Optional

from pydantic import BaseModel, Field

class LLMInput(BaseModel):
    prompt: str = Field("xin chào")
    max_length: Optional[int] = Field(128)

class LLMOutput(BaseModel):
    generate_text: str = Field(
        default="",
        description="text generated by the model"
    )
    error: Optional[Any] = Field(
        default=None,  # Set default to None
        description="Error information if an exception occurs"
    )
