# routes/speech.py
import asyncio
import json
import logging
from typing import Annotated, List, Literal, Self, Optional

from fastapi import APIRouter, HTTPException, Depends # Keep original imports
from fastapi.responses import StreamingResponse
# Keep Pydantic model for the request body to define the API spec
from pydantic import BaseModel, Field, field_validator, model_validator

# Import dependencies and helpers
from dependencies import (
    ConfigDependency, F5ttsModelManagerDependency, VocoderDependency,
    get_custom_voice_ref_path, get_model_class_from_name, list_custom_voice_ids # Add voice lookup
)
from f5tts_utils import generate_audio_f5tts # Correct function name
from audio import convert_audio_format
from model_aliases import ModelId # Keep if used for validation/aliasing
from api_types import ResponseFormat, SUPPORTED_RESPONSE_FORMATS, Voice

logger = logging.getLogger(__name__)
router = APIRouter(tags=["text-to-speech"]) # Corrected tag

# --- Original Request Body Definition (Keep for API spec) ---
class CreateSpeechRequestBody(BaseModel):
    model: str = Field(default="tts-1", description="Model ID to use (e.g., 'tts-1', 'F5-TTS_v1', 'custom_piper_model')")
    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(default="alloy", description="Voice ID to use (e.g., 'alloy', 'my_custom_voice_1').")
    response_format: ResponseFormat = Field(default="mp3")
    speed: float = Field(default=1.0, ge=0.25, le=4.0) # OpenAI's range
    # sample_rate: int | None = Field(None) # OpenAI doesn't have sample_rate param

    # Remove F5/Piper specific validations from here, handle in endpoint logic
    # @model_validator(mode="after") ...

# --- Modified Synthesis Endpoint ---
@router.post("/v1/audio/speech")
async def synthesize_speech(
    body: CreateSpeechRequestBody, # Use the Pydantic model
    config: ConfigDependency,
    f5tts_model_manager: F5ttsModelManagerDependency, # Inject if needed
    vocoder: VocoderDependency,               # Inject if needed
    # Inject other TTS engine dependencies if you have them (e.g., Piper client)
):
    logger.info(f"Received speech request: model='{body.model}', voice='{body.voice}'")

    # --- Routing Logic ---

    # 1. Check if it's an F5-TTS request
    if body.model in config.f5tts.f5_engine_model_ids:
        logger.info(f"Handling request with F5-TTS engine for model '{body.model}'.")

        # Find the reference audio for the custom voice ID
        custom_ref_path = get_custom_voice_ref_path(config, body.voice)

        if not custom_ref_path:
            logger.warning(f"Custom voice ID '{body.voice}' not found in storage for F5-TTS request.")
            raise HTTPException(status_code=400, detail=f"Voice '{body.voice}' is not a valid custom voice for model '{body.model}'.")

        # Get F5-TTS model definition from config
        model_def = config.f5tts.model_definitions.get(body.model)
        if not model_def: # Should not happen if f5_engine_model_ids is consistent
             logger.error(f"Internal config error: Model ID '{body.model}' listed in f5_engine_model_ids but has no definition.")
             raise HTTPException(status_code=500, detail="Server configuration error for the requested model.")

        try:
            model_cls = get_model_class_from_name(model_def.model_class_name)
            model_cfg_dict = json.loads(model_def.config_json)

            model_wrapper = f5tts_model_manager.load_model(
                model_cls=model_cls,
                ckpt_path=model_def.ckpt_path,
                model_cfg=model_cfg_dict,
                vocab_path=model_def.vocab_path,
            )
        except Exception as e:
             logger.error(f"Failed to initiate F5-TTS model loading for '{body.model}': {e}", exc_info=True)
             raise HTTPException(status_code=500, detail="Failed to load TTS model.")

        # Define the async generator for F5-TTS
        async def f5_stream_generator():
            try:
                with model_wrapper as loaded_ema_model:
                    # Adjust parameters based on request body (mapping OpenAI speed to F5 speed if needed)
                    # F5 speed seems to be handled differently, maybe map 1.0 -> 1.0, faster/slower needs experiment
                    f5_speed = body.speed # Direct mapping for now, adjust if needed
                    # TODO: Get NFE, cross-fade, remove_silence from config or defaults? OpenAI body doesn't have them.
                    nfe_step = 32 # Example default
                    cross_fade_duration = 0.15 # Example default
                    remove_silence = False # Example default
                    # OpenAI body doesn't have output sample rate, use model default or config? Let's use default for now.
                    output_sr = None # Let generate_audio_f5tts handle it

                    logger.info(f"Generating F5-TTS audio using ref: {custom_ref_path}")
                    audio_stream = generate_audio_f5tts(
                        ema_model=loaded_ema_model,
                        vocoder=vocoder,
                        ref_audio_path=custom_ref_path, # Use the stored reference!
                        gen_text=body.input,
                        ref_text=None, # Custom voices usually won't have stored ref_text
                        speed=f5_speed,
                        nfe_step=nfe_step,
                        cross_fade_duration=cross_fade_duration,
                        output_sample_rate=output_sr, # Let model decide internal, convert later if needed
                        remove_silence=remove_silence,
                    )

                    # Get the actual sample rate AFTER generation for conversion
                    # This requires modifying generate_audio_f5tts or getting it from the model/vocoder
                    # Placeholder: Assume 24kHz which is common for these models
                    # TODO: Fix this assumption!
                    internal_sample_rate = 24000

                    async for audio_chunk in audio_stream:
                        if body.response_format != "pcm":
                            converted_chunk = await asyncio.to_thread(
                                convert_audio_format,
                                audio_chunk,
                                internal_sample_rate, # Use the correct internal rate
                                body.response_format
                            )
                            yield converted_chunk
                        else:
                            yield audio_chunk
                    logger.info("F5-TTS audio generation finished.")
            except Exception as e:
                logger.error(f"Error during F5-TTS generation stream: {e}", exc_info=True)
                # Don't raise HTTPException here, let FastAPI handle generator errors
                # Maybe yield an error message or just stop? For now, just log.

        return StreamingResponse(f5_stream_generator(), media_type=f"audio/{body.response_format}")

    # 2. Check if it's a request for another engine (e.g., Piper, OpenAI Proxy)
    # elif body.model.startswith("piper-"):
    #     # ... handle Piper TTS ...
    #     logger.info(f"Routing to Piper TTS for model '{body.model}'.")
    #     raise HTTPException(status_code=501, detail="Piper TTS not implemented.")
    # elif body.model in ["tts-1", "tts-1-hd"]:
    #      # ... handle proxying to actual OpenAI ...
    #      logger.info(f"Routing to OpenAI TTS API for model '{body.model}'.")
    #      raise HTTPException(status_code=501, detail="OpenAI TTS proxy not implemented.")

    # 3. If model/voice combination doesn't match any known engine/custom voice
    else:
        logger.warning(f"Model '{body.model}' and voice '{body.voice}' combination is not supported.")
        raise HTTPException(status_code=400, detail=f"The requested model '{body.model}' and voice '{body.voice}' combination is not supported.")
    




# --- Add back the list voices endpoint ---
@router.get("/v1/audio/speech/voices")
def list_voices(
    config: ConfigDependency,
    # Optional: Filter by model ID if you want to list voices only for a specific engine
    model_id: ModelId | None = None
) -> List[Voice]: # Return a list of Voice objects
    """Lists available voices, potentially filtered by model ID."""
    voices: List[Voice] = []
    logger.info(f"Request to list voices, model filter: {model_id}")

    # --- Add Custom F5-TTS Voices ---
    # Check if the filter matches any F5 engine ID OR if there's no filter
    should_add_f5 = model_id is None or model_id in config.f5tts.f5_engine_model_ids
    if should_add_f5:
        logger.debug("Adding custom F5-TTS voices...")
        custom_voice_ids = list_custom_voice_ids(config)
        for voice_id_str in custom_voice_ids:
            # Create a Voice object for each custom voice ID found
            voices.append(Voice(
                id=voice_id_str,
                name=voice_id_str # Use ID as name for simplicity, or add metadata storage
                # Add other fields if your Voice model requires them
                # engine="f5tts-custom",
                # associated_model=config.f5tts.f5_engine_model_ids # Could list compatible models
            ))
        logger.debug(f"Added {len(custom_voice_ids)} custom F5-TTS voices.")


    logger.info(f"Returning {len(voices)} voices matching filter.")
    return voices
