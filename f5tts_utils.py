#f5tts_utils.py
import asyncio
import logging
import tempfile
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, List, Optional, Union, Type # Added Type

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .api_types import F5TTSModel
from .config import F5TTSModelDefinition

# --- Assumed Imports from your F5-TTS setup ---
# Make sure these paths are correct for your project structure
try:
    from f5_tts.model import DiT, UNetT
    from f5_tts.infer.utils_infer import (
        infer_process,
        preprocess_ref_audio_text,
        remove_silence_for_generated_wav,
    )
    # Define the compatible model type
    F5CompatibleModel = Union[DiT, UNetT]
except ImportError as e:
    raise ImportError(f"Could not import F5-TTS components. Ensure f5_tts package is installed and accessible: {e}")

# --- Assumed Import from speaches ---
# Make sure this path is correct
try:
    from .audio import resample_audio # As used in the Kokoro example
except ImportError:
    # Define a placeholder if running standalone
    def resample_audio(audio_bytes: bytes, input_rate: int, output_rate: int) -> bytes:
        print(f"Placeholder: Resampling needed from {input_rate} to {output_rate}. Implement `resample_audio`.")
        # Basic placeholder: Requires a real implementation using e.g., torchaudio.transforms.Resample
        if input_rate == output_rate:
            return audio_bytes
        # In a real scenario, convert bytes->tensor, resample, convert tensor->bytes
        raise NotImplementedError("Audio resampling function is not implemented.")
    print("Warning: Using placeholder `resample_audio` function.")



logger = logging.getLogger(__name__)





async def generate_audio_f5tts(
    ema_model: F5CompatibleModel,
    vocoder: Any, # Replace Any with the specific vocoder class type if known
    ref_audio_path: Union[str, Path],
    gen_text: str,
    *, # Keyword-only arguments follow
    ref_text: Optional[str] = None,
    speed: float = 1.0,
    nfe_step: int = 32,
    cross_fade_duration: float = 0.15,
    output_sample_rate: Optional[int] = None, # Target sample rate (like Kokoro example)
    remove_silence: bool = False,
) -> AsyncGenerator[bytes, None]:
    """
    Generates audio using an F5-TTS compatible model and yields the audio data as bytes.

    Note: Unlike the Kokoro streaming example, infer_process currently generates
          the entire audio clip at once. This function mimics the async generator
          signature but will likely yield the full audio in a single chunk after
          synchronous processing. True streaming would require modifying infer_process.

    Args:
        ema_model: The loaded F5-TTS or E2-TTS model instance.
        vocoder: The loaded vocoder instance.
        ref_audio_path: Path to the reference audio file.
        gen_text: The text to synthesize.
        ref_text: Optional transcript of the reference audio. If None, it will be transcribed.
        speed: Speed adjustment factor for synthesis.
        nfe_step: Number of Function Evaluations (denoising steps).
        cross_fade_duration: Duration in seconds for cross-fading between chunks.
        output_sample_rate: If specified, resample the output audio to this rate.
        remove_silence: If True, attempt to remove leading/trailing silences.

    Yields:
        bytes: Chunk(s) of raw PCM audio data (int16 format). Currently yields one chunk.
    """
    start_time = time.perf_counter()
    logger.info(f"Starting F5-TTS generation for {len(gen_text)} characters. Speed={speed}, NFE={nfe_step}")

    # --- 1. Preprocessing Reference ---
    try:
        # Use logger instead of Gradio info/warning functions
        def log_info(msg): logger.info(f"Preprocessing: {msg}")
        # def log_warning(msg): logger.warning(f"Preprocessing: {msg}") # Add if preprocess uses warnings

        ref_audio_path_obj = Path(ref_audio_path)
        if not ref_audio_path_obj.is_file():
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")

        logger.debug("Preprocessing reference audio and text...")
        # Note: preprocess_ref_audio_text performs loading, resampling, VAD, and optional transcription
        ref_audio_processed_tensor, ref_text_processed = preprocess_ref_audio_text(
            str(ref_audio_path_obj), # Function expects string path
            ref_text,
            show_info=log_info,
            # show_warning=log_warning, # Add if available/needed
        )
        logger.debug("Reference preprocessing complete.")

    except FileNotFoundError as e:
        logger.error(f"Reference audio file error: {e}")
        raise # Re-raise critical error
    except Exception as e:
        logger.error(f"Error during reference audio/text preprocessing: {e}", exc_info=True)
        # Depending on policy, you might want to raise or return/yield nothing
        raise RuntimeError(f"Preprocessing failed: {e}") from e

    # --- 2. Core TTS Inference ---
    # Note: infer_process is likely synchronous. For truly non-blocking async,
    # this should be run in a separate thread using asyncio.to_thread.
    final_wave_np: Optional[np.ndarray] = None
    final_sample_rate: Optional[int] = None
    try:
        logger.debug("Calling synchronous infer_process...")
        # Running the synchronous function directly in the async function
        # For better async performance, wrap this call:
        # final_wave_np, final_sample_rate, _ = await asyncio.to_thread(
        #      infer_process, ... )

        final_wave_np, final_sample_rate, _combined_spectrogram = infer_process(
            ref_audio=ref_audio_processed_tensor,
            ref_text=ref_text_processed,
            gen_text=gen_text,
            ema_model=ema_model,
            vocoder=vocoder,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            show_info=log_info, # Use logger for internal info messages
            progress=None,      # No Gradio progress bar here
        )
        logger.debug(f"infer_process completed. Original SR: {final_sample_rate}, shape: {final_wave_np.shape}")

    except Exception as e:
        logger.error(f"Error during F5-TTS infer_process: {e}", exc_info=True)
        raise RuntimeError(f"TTS Inference failed: {e}") from e # Re-raise as runtime error


    # Ensure we got valid output
    if final_wave_np is None or final_sample_rate is None:
         logger.error("infer_process did not return a valid waveform or sample rate.")
         raise RuntimeError("TTS Inference failed to produce output.")


    # --- 3. Optional Silence Removal ---
    if remove_silence:
        logger.info("Applying silence removal...")
        temp_wav_path = None
        try:
            # Use a temporary file for soundfile write -> torchaudio read
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                temp_wav_path = tmp_f.name
            sf.write(temp_wav_path, final_wave_np, final_sample_rate)

            # This function modifies the file in-place
            remove_silence_for_generated_wav(temp_wav_path)

            # Reload the processed audio
            final_wave_tensor, sr_after_silence = torchaudio.load(temp_wav_path)
            if sr_after_silence != final_sample_rate:
                 logger.warning(f"Sample rate changed after silence removal ({final_sample_rate} -> {sr_after_silence}). Using new rate.")
                 final_sample_rate = sr_after_silence
            # Ensure numpy array float32 for consistency
            final_wave_np = final_wave_tensor.squeeze().cpu().numpy().astype(np.float32)
            logger.info(f"Silence removal applied. New shape: {final_wave_np.shape}")

        except Exception as e:
            logger.error(f"Error during silence removal: {e}", exc_info=True)
            # Decide whether to raise or just warn and continue
            logger.warning("Proceeding without silence removal due to error.")
        finally:
            # Clean up the temporary file
            if temp_wav_path and Path(temp_wav_path).exists():
                try:
                    Path(temp_wav_path).unlink()
                    logger.debug(f"Deleted temporary silence removal file: {temp_wav_path}")
                except OSError as e:
                    logger.error(f"Failed to delete temporary file {temp_wav_path}: {e}")


    # --- 4. Format Conversion (float32 -> int16 bytes) ---
    if not isinstance(final_wave_np, np.ndarray):
        logger.error(f"Expected numpy array after processing, got {type(final_wave_np)}. Cannot proceed.")
        raise TypeError("Invalid waveform type after processing.")

    if final_wave_np.dtype != np.float32:
        logger.warning(f"Waveform dtype is {final_wave_np.dtype}, expected float32. Converting.")
        final_wave_np = final_wave_np.astype(np.float32)

    # Scale to int16 range and convert type
    int16_audio = (final_wave_np * np.iinfo(np.int16).max).astype(np.int16)
    audio_bytes = int16_audio.tobytes()


    # --- 5. Optional Resampling ---
    current_sample_rate = final_sample_rate
    if output_sample_rate is not None and output_sample_rate != current_sample_rate:
        logger.info(f"Resampling audio from {current_sample_rate} Hz to {output_sample_rate} Hz...")
        try:
            resampled_bytes = await asyncio.to_thread( # Use to_thread if resample_audio is sync
                 resample_audio,
                 audio_bytes,
                 current_sample_rate,
                 output_sample_rate
            )
            # Or if resample_audio is async:
            # resampled_bytes = await resample_audio(...)
            audio_bytes = resampled_bytes
            current_sample_rate = output_sample_rate # Update effective sample rate
            logger.info("Resampling complete.")
        except Exception as e:
            logger.error(f"Error during resampling: {e}", exc_info=True)
            logger.warning("Proceeding with original sample rate due to resampling error.")


    # --- 6. Yield Result ---
    # Currently yields the entire audio as one chunk because infer_process is not streaming.
    duration = time.perf_counter() - start_time
    logger.info(f"Finished F5-TTS generation. Audio duration: {len(int16_audio)/current_sample_rate:.2f}s. Total time: {duration:.2f}s")
    yield audio_bytes




def create_f5tts_model_api_object(
    model_id: str,
    definition: 'F5TTSModelDefinition', # Forward reference if F5TTSModelDefinition is in config.py
    # You might pass the whole Config object if needed for more context
) -> F5TTSModel:
    """Creates an F5TTSModel API object from a configuration definition."""
    # Determine owner (simple heuristic based on path or ID)
    owner = "unknown"
    if definition.ckpt_path.startswith("hf://"):
        parts = definition.ckpt_path.split('/')
        if len(parts) >= 4: # e.g., hf://Owner/Repo/file.pt
            owner = parts[2]
    elif '/' in model_id: # e.g., Owner/ModelName as ID
        owner = model_id.split('/')[0]

    # TODO: Enhance language detection/configuration if needed
    languages = ["en", "zh"] # Default for now

    # TODO: Enhance creation time fetching if needed (e.g., from HF Hub)
    creation_time = 0

    return F5TTSModel(
        id=model_id,
        created=creation_time,
        owned_by=owner,
        language=languages,
        task="text-to-speech",
        architecture=definition.model_class_name, # DiT or UNetT from config
    )