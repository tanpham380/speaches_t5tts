# routes/models.py
import os
import logging
from typing import Union, List, Optional # Added Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
)

# Import configuration and dependency
from dependencies import ConfigDependency

# Import utility modules and helpers
import whisper_utils
from f5tts_utils import create_f5tts_model_api_object

# Import your API types
from api_types import (
    ListModelsResponse,
    Model,
    F5TTSModel,
    ModelTask,
)
from model_aliases import ModelId

router = APIRouter(tags=["models"])
logger = logging.getLogger(__name__)


@router.get("/v1/models")
def get_models(
    config: ConfigDependency,
    task: ModelTask | None = None
) -> ListModelsResponse:
    """Lists available models, optionally filtered by task."""
    models: List[Union[Model, F5TTSModel]] = []
    logger.info(f"Request to list models, task filter: {task}")
    # print
    if task == "-":
        logger.warning("Task filter is '-', returning empty model list.")
        return ListModelsResponse(data=models)

    # --- Add F5-TTS Models ---
    if task is None or task == "text-to-speech":
        logger.debug("Adding configured F5-TTS/E2-TTS models...")
        try:
            count = 0
            for model_id, definition in config.f5tts.model_definitions.items():
                if model_id in config.f5tts.f5_engine_model_ids:
                    f5_model_obj = create_f5tts_model_api_object(model_id, definition)
                    models.append(f5_model_obj)
                    count += 1
            logger.debug(f"Added {count} F5-TTS/E2-TTS models from configuration.")
        except Exception as e:
            logger.error(f"Failed to process F5-TTS model definitions from config: {e}", exc_info=True)

    # --- Add Whisper Models ---
    if task is None or task == "automatic-speech-recognition":
        logger.debug("Adding ASR (Whisper) models...")
        try:
            whisper_list_func = (
                whisper_utils.list_local_whisper_models
                if os.getenv("HF_HUB_OFFLINE") is not None
                else whisper_utils.list_whisper_models
            )
            logger.debug(f"Using {'local' if os.getenv('HF_HUB_OFFLINE') else 'online'} Whisper model listing.")

            whisper_models_gen = whisper_list_func()
            count = 0
            for w_model in whisper_models_gen:
                 models.append(w_model)
                 count += 1
            logger.debug(f"Added {count} Whisper models.")
        except Exception as e:
            logger.error(f"Failed to get Whisper models: {e}", exc_info=True)

    # --- Return combined list ---
    logger.info(f"Returning {len(models)} models matching filter.")
    return ListModelsResponse(data=models)


# --- Updated Get Model Details ---
@router.get("/v1/models/{model_id:path}")
def get_model(
    model_id: ModelId, # Use the alias-resolved ID
    config: ConfigDependency
) -> Union[Model, F5TTSModel]:
    """Retrieves details for a specific model by its ID."""
    logger.info(f"Request to get model details for ID: {model_id}")

    # 1. Check F5-TTS models from config first
    if model_id in config.f5tts.f5_engine_model_ids:
        definition = config.f5tts.model_definitions.get(model_id)
        if definition:
            logger.info(f"Model ID '{model_id}' found in F5-TTS configuration.")
            f5_model_obj = create_f5tts_model_api_object(model_id, definition)
            return f5_model_obj
        else:
            logger.error(f"Internal config error: Model ID '{model_id}' in f5_engine_model_ids but missing definition.")
            raise HTTPException(status_code=500, detail="Server configuration error for this model ID.")

    # 2. Check Whisper models using specific lookup functions
    logger.debug(f"Model ID '{model_id}' not F5-TTS, checking Whisper models...")
    try:
        whisper_model: Optional[Model] = None
        get_whisper_info_func = (
            whisper_utils.get_local_whisper_model_info
            if os.getenv("HF_HUB_OFFLINE") is not None
            else whisper_utils.get_hf_whisper_model_info
        )
        logger.debug(f"Using {'local' if os.getenv('HF_HUB_OFFLINE') else 'online'} Whisper model info retrieval.")
        whisper_model = get_whisper_info_func(model_id)

        if whisper_model:
            logger.info(f"Model ID '{model_id}' found as a Whisper model.")
            return whisper_model
        else:
             logger.debug(f"Model ID '{model_id}' not found among Whisper models via specific lookup.")

    except Exception as e:
        logger.error(f"Error checking Whisper models for ID '{model_id}': {e}", exc_info=True)
        # Don't fail request yet, maybe it's another type

    # 3. Add checks for other model types if necessary

    # 4. If not found anywhere
    logger.warning(f"Model ID '{model_id}' not found in any known configuration or registry.")
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")