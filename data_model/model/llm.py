from typing import Any, Optional

from pydantic import BaseModel, Field


class TextOutput(BaseModel):
    text: str = Field("")

class SpeedOutput(BaseModel):
    tokens_per_second: int = Field(
        default=0,
        description="tokens per second"
    )
    avg_time_per_iteration: int = Field(
        default=0,
        description="avg time per iteration (seconds)"
    )