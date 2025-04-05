# model_aliases.py
from functools import lru_cache
import json
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field




ModelId = Annotated[
    str,
    # BeforeValidator(resolve_model_id_alias),
    Field(
        min_length=1,
        description="The ID of the model. You can get a list of available models by calling `/v1/models`.",
        examples=[
            "Systran/faster-distil-whisper-large-v3",
            "bofenghuang/whisper-large-v2-cv11-french-ct2",
            "openai/whisper-large-v2",
            "erax-ai/EraX-WoW-Turbo-V1.1-CT2"
        ],
    ),
]
