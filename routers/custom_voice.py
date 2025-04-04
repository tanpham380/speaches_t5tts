# routers/custom_voices.py
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any

from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File, Form, status
)
from pydantic import BaseModel

from speaches.dependencies import ConfigDependency, ApiKeyDependency # Assuming ApiKeyDependency handles auth
from speaches.dependencies import save_upload_file_tmp, list_custom_voice_ids, get_custom_voice_ref_path

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/v1/voices/custom", # Or "/admin/voices"
    tags=["custom-voice-management"],
    # dependencies=[ApiKeyDependency] # Apply auth to all routes in this router
)

# --- Pydantic model for response ---
class CustomVoiceInfo(BaseModel):
    voice_id: str
    reference_file_path: str | None = None # Path relative to server storage for info

class ListCustomVoicesResponse(BaseModel):
    voices: List[CustomVoiceInfo]

# --- Endpoint to create a new voice ---
@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=CustomVoiceInfo,
    # dependencies=[ApiKeyDependency] # Can add per-route auth too
)
async def create_custom_voice(
    config: ConfigDependency,
    voice_id: str = Form(..., description="Unique ID for the new voice (e.g., 'user_x_voice_1'). Avoid special characters."),
    reference_audio: UploadFile = File(..., description="Reference audio file (.wav, .mp3, etc.) for cloning."),
    # Add optional metadata if needed: description: str = Form(None)
):
    """Creates a new custom voice by uploading reference audio."""
    logger.info(f"Request to create custom voice with ID: {voice_id}")

    # Basic sanitization
    if ".." in voice_id or "/" in voice_id or "\\" in voice_id:
        raise HTTPException(status_code=400, detail="Invalid characters in voice_id.")

    voice_dir = config.custom_voices.storage_path / voice_id
    if voice_dir.exists():
        raise HTTPException(status_code=409, detail=f"Voice ID '{voice_id}' already exists.")

    temp_audio_path: Path | None = None
    try:
        # Save uploaded file temporarily first for validation/processing if needed
        temp_audio_path = await save_upload_file_tmp(reference_audio)

        # TODO: Add optional validation here (e.g., check audio format/duration using ffprobe or soundfile)
        # Example: Check if it's readable
        # import soundfile as sf
        # try:
        #     sf.read(temp_audio_path)
        # except Exception as e:
        #     raise HTTPException(status_code=400, detail=f"Invalid reference audio file: {e}")

        # Create the final directory and move the file
        voice_dir.mkdir(parents=True)
        # Use a consistent filename like 'reference' + original extension
        target_filename = f"reference{temp_audio_path.suffix}"
        final_path = voice_dir / target_filename
        shutil.move(str(temp_audio_path), final_path)
        logger.info(f"Successfully created custom voice '{voice_id}' with reference: {final_path}")
        return CustomVoiceInfo(voice_id=voice_id, reference_file_path=str(final_path))

    except HTTPException:
        raise # Re-raise validation errors etc.
    except Exception as e:
        logger.error(f"Failed to create custom voice '{voice_id}': {e}", exc_info=True)
        # Clean up partial directory if created
        if voice_dir.exists():
            shutil.rmtree(voice_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Failed to create custom voice.")
    finally:
        # Clean up temp file if it wasn't moved
        if temp_audio_path and temp_audio_path.exists():
            try:
                temp_audio_path.unlink()
            except OSError: pass

# --- Endpoint to list custom voices ---
@router.get("", response_model=ListCustomVoicesResponse)
async def list_managed_custom_voices(config: ConfigDependency):
    """Lists all created custom voice IDs."""
    voice_ids = list_custom_voice_ids(config)
    voices_info = [CustomVoiceInfo(voice_id=vid) for vid in voice_ids]
    # Optionally, add logic here to find the actual file path for each voice if needed
    return ListCustomVoicesResponse(voices=voices_info)

# --- Endpoint to delete a custom voice ---
@router.delete("/{voice_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_custom_voice(voice_id: str, config: ConfigDependency):
    """Deletes a custom voice and its reference audio."""
    logger.info(f"Request to delete custom voice: {voice_id}")
    voice_dir = config.custom_voices.storage_path / voice_id

    if not voice_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Custom voice '{voice_id}' not found.")

    try:
        shutil.rmtree(voice_dir)
        logger.info(f"Successfully deleted custom voice directory: {voice_dir}")
    except Exception as e:
        logger.error(f"Failed to delete custom voice directory '{voice_dir}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete custom voice.")

    # Return No Content