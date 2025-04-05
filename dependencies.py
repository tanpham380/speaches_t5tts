# dependancies.py
from functools import lru_cache
import logging
from typing import Annotated, Any, List, Optional, Type, Union
import asyncio # Added
import tempfile # Added
from pathlib import Path # Added

import av.error
from cached_path import cached_path
from fastapi import (
    Depends,
    Form,
    HTTPException,
    UploadFile,
    status,
    File,
    Form,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper.audio import decode_audio
from httpx import ASGITransport, AsyncClient
from numpy import float32
from numpy.typing import NDArray
from openai import AsyncOpenAI
from openai.resources.audio import AsyncSpeech, AsyncTranscriptions
from openai.resources.chat.completions import AsyncCompletions

from config import Config
from model_manager import F5TTSModelManager,  WhisperModelManager
from f5_tts.infer.utils_infer import load_vocoder
from f5_tts.model import DiT, UNetT # Import model classes



logger = logging.getLogger(__name__)

# NOTE: `get_config` is called directly instead of using sub-dependencies so that these functions could be used outside of `FastAPI`


# https://fastapi.tiangolo.com/advanced/settings/?h=setti#creating-the-settings-only-once-with-lru_cache
# WARN: Any new module that ends up calling this function directly (not through `FastAPI` dependency injection) should be patched in `tests/conftest.py`
@lru_cache
def get_config() -> Config:
    return Config()


ConfigDependency = Annotated[Config, Depends(get_config)]


@lru_cache
def get_model_manager() -> WhisperModelManager:
    config = get_config()
    return WhisperModelManager(config.whisper)


ModelManagerDependency = Annotated[WhisperModelManager, Depends(get_model_manager)]


# --- Vocoder Dependency ---
@lru_cache
def get_vocoder(config: Config = Depends(get_config)) -> Any: # Replace Any with actual type
    logger.info("Loading vocoder...")
    # Assuming load_vocoder takes the path from config
    # And handles device placement internally
    try:
        # Resolve path similarly to how F5TTSModelManager does
        vocoder_path = config.f5tts.vocoder_path
        if vocoder_path.startswith("hf://"):
            resolved_path = str(cached_path(vocoder_path))
            logger.info(f"Resolved vocoder path '{vocoder_path}' to '{resolved_path}'")
        else:
            resolved_path = vocoder_path
            logger.info(f"Using local vocoder path '{resolved_path}'")

        if not Path(resolved_path).exists():
             raise FileNotFoundError(f"Vocoder file not found: {resolved_path}")

        vocoder_instance = load_vocoder(model_path=resolved_path) # Adjust arg name if needed
        logger.info("Vocoder loaded successfully.")
        return vocoder_instance
    except Exception as e:
        logger.error(f"Failed to load vocoder from {config.f5tts.vocoder_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load vocoder model.")

VocoderDependency = Annotated[Any, Depends(get_vocoder)] # Replace Any

# --- Update F5TTS Manager Dependency ---
@lru_cache
def get_t5tts_model_manager() -> F5TTSModelManager:
    config = get_config()
    # Use the specific TTL from F5TTS config
    return F5TTSModelManager(config.f5tts.ttl)

F5ttsModelManagerDependency = Annotated[F5TTSModelManager, Depends(get_t5tts_model_manager)]

# --- Helper for Reference Audio ---
async def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    """Saves UploadFile to a temporary file and returns the path."""
    try:
        # Create a temporary file with the correct suffix
        suffix = Path(upload_file.filename).suffix if upload_file.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            # Read chunks from the upload file and write to the temp file
            while content := await upload_file.read(1024 * 1024): # Read in 1MB chunks
                 tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
            logger.info(f"Saved uploaded file '{upload_file.filename}' to temporary path: {tmp_path}")
            return tmp_path
    except Exception as e:
        logger.error(f"Failed to save uploaded file '{upload_file.filename}' to temporary file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not save reference audio.")
    finally:
        # Ensure the file pointer is closed, releasing the resource
        if upload_file:
            await upload_file.close()

def get_model_class_from_name(class_name: str) -> Type[Union[DiT, UNetT]]:
     if class_name == "DiT":
         return DiT
     elif class_name == "UNetT":
         return UNetT
     else:
         raise ValueError(f"Unknown model class name: {class_name}")


# --- Custom Voice Helper ---
def get_custom_voice_ref_path(config: Config, voice_id: str) -> Optional[Path]:
    """
    Constructs the expected path for a custom voice's reference audio.
    Returns the Path object if it exists, otherwise None.
    """
    # Basic sanitization to prevent directory traversal
    if ".." in voice_id or "/" in voice_id or "\\" in voice_id:
        logger.warning(f"Invalid characters found in custom voice_id: {voice_id}")
        return None

    # Assuming storage structure: storage_path / voice_id / reference.wav (or .mp3 etc.)
    # You might need a more robust way to find the exact file if extensions vary.
    voice_dir = config.custom_voices.storage_path / voice_id
    if not voice_dir.is_dir():
        return None

    # Look for common audio file extensions
    possible_extensions = [".wav", ".mp3", ".flac", ".ogg"]
    for ext in possible_extensions:
        ref_file = voice_dir / f"reference{ext}"
        if ref_file.is_file():
            logger.debug(f"Found custom voice reference file: {ref_file}")
            return ref_file

    logger.warning(f"No reference audio file found in directory for custom voice_id: {voice_id}")
    return None

def list_custom_voice_ids(config: Config) -> List[str]:
    """Lists the IDs of custom voices found in the storage path."""
    if not config.custom_voices.storage_path.exists():
        return []
    return [d.name for d in config.custom_voices.storage_path.iterdir() if d.is_dir()]



security = HTTPBearer()


async def verify_api_key(
    config: ConfigDependency, credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> None:
    if credentials.credentials != config.api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


ApiKeyDependency = Depends(verify_api_key)



@lru_cache
def get_completion_client() -> AsyncCompletions:
    config = get_config()
    oai_client = AsyncOpenAI(
        base_url=config.chat_completion_base_url,
        api_key=config.chat_completion_api_key.get_secret_value(),
        max_retries=1,
    )
    return oai_client.chat.completions


CompletionClientDependency = Annotated[AsyncCompletions, Depends(get_completion_client)]


@lru_cache
def get_speech_client() -> AsyncSpeech:
    # this might not work as expected if `speech_router` won't have shared state (access to the same `model_manager`) with the main FastAPI `app`. TODO: verify
    from routers.speech import (
        router as speech_router,
    )

    config = get_config()
    http_client = AsyncClient(
        transport=ASGITransport(speech_router), base_url="http://test/v1"
    )  # NOTE: "test" can be replaced with any other value
    oai_client = AsyncOpenAI(
        http_client=http_client,
        api_key=config.api_key.get_secret_value() if config.api_key else "cant-be-empty",
        max_retries=1,
    )
    return oai_client.audio.speech


SpeechClientDependency = Annotated[AsyncSpeech, Depends(get_speech_client)]


@lru_cache
def get_transcription_client() -> AsyncTranscriptions:
    # this might not work as expected if `stt_router` won't have shared state (access to the same `model_manager`) with the main FastAPI `app`. TODO: verify
    from routers.stt import (
        router as stt_router,
    )

    config = get_config()
    http_client = AsyncClient(
        transport=ASGITransport(stt_router), base_url="http://test/v1"
    )  # NOTE: "test" can be replaced with any other value
    oai_client = AsyncOpenAI(
        http_client=http_client,
        api_key=config.api_key.get_secret_value() if config.api_key else "cant-be-empty",
        max_retries=1,
    )
    return oai_client.audio.transcriptions


TranscriptionClientDependency = Annotated[AsyncTranscriptions, Depends(get_transcription_client)]
