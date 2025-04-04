# config.py
import json
from typing import Literal, Dict, Any, Optional, List , Type

from pathlib import Path # Added Path
from pydantic import BaseModel, Field, SecretStr, DirectoryPath, model_validator # Added DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Existing Config Parts ---
SAMPLES_PER_SECOND = 16000
SAMPLE_WIDTH = 2
BYTES_PER_SECOND = SAMPLES_PER_SECOND * SAMPLE_WIDTH
type Device = Literal["cpu", "cuda", "auto"]
type Quantization = Literal[
    "int8", "int8_float16", "int8_bfloat16", "int8_float32", "int16", "float16", "bfloat16", "float32", "default"
]

class WhisperConfig(BaseModel):
    inference_device: Device = "auto"
    device_index: int | list[int] = 0
    compute_type: Quantization = "default"
    cpu_threads: int = 0
    num_workers: int = 1
    ttl: int = Field(default=300, ge=-1)
    use_batched_mode: bool = False

# --- F5-TTS Specific Config ---
DEFAULT_F5_TTS_MODEL_ID = "F5-TTS_v1" # Use simpler IDs for user requests
DEFAULT_E2_TTS_MODEL_ID = "E2-TTS"

class F5TTSModelDefinition(BaseModel):
    model_class_name: Literal["DiT", "UNetT"]
    ckpt_path: str # Can be local or hf://
    vocab_path: Optional[str] = None # Can be local or hf://
    config_json: str # JSON string representing the config dict

class F5TTSConfig(BaseModel):
    ttl: int = Field(default=300, ge=-1, description="TTL for F5-TTS models in seconds.")
    vocoder_path: str = Field(default="hf://SWivid/BigVGAN/bigvgan_base_24khz_100band.pt", description="Path (local or hf://) to the vocoder model.")
    # Maps user-facing model IDs to their technical definitions
    model_definitions: Dict[str, F5TTSModelDefinition] = {
        DEFAULT_F5_TTS_MODEL_ID: F5TTSModelDefinition(
            model_class_name="DiT",
            ckpt_path="hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
            vocab_path="hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
            config_json=json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
        ),
        DEFAULT_E2_TTS_MODEL_ID: F5TTSModelDefinition(
             model_class_name="UNetT",
             ckpt_path="hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors",
             vocab_path=None,
             config_json=json.dumps(dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)),
        )
        # Add other custom F5/E2 model definitions here if needed
    }
    # Identifies which model IDs should trigger F5-TTS logic
    f5_engine_model_ids: List[str] = Field(default=[DEFAULT_F5_TTS_MODEL_ID, DEFAULT_E2_TTS_MODEL_ID])

# --- Custom Voice Storage Config ---
class CustomVoicesConfig(BaseModel):
    storage_path: DirectoryPath = Field(default=Path("./custom_voices"), description="Directory to store custom voice reference audio files.")

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")
    api_key: SecretStr | None = None
    log_level: str = "info" # Changed default to info
    host: str = Field(alias="UVICORN_HOST", default="0.0.0.0")
    port: int = Field(alias="UVICORN_PORT", default=8000)
    allow_origins: list[str] | None = None
    whisper: WhisperConfig = WhisperConfig()
    f5tts: F5TTSConfig = F5TTSConfig()
    custom_voices: CustomVoicesConfig = CustomVoicesConfig() 

    @model_validator(mode='after')
    def check_paths(self) -> 'Config':
        self.custom_voices.storage_path.mkdir(parents=True, exist_ok=True)
        return self