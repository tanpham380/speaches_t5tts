# routers/stt.py

import asyncio
from collections.abc import Generator, Iterable
import logging
from typing import Annotated, AsyncGenerator, Literal, List
import av

# --- Imports ---
import numpy as np
from numpy.typing import NDArray
# --- End Imports ---

from fastapi import (
    APIRouter,
    Form,
    Request,
    Response,
    HTTPException,
    UploadFile,
    File,
    Depends
)
from fastapi.responses import StreamingResponse
from faster_whisper.transcribe import BatchedInferencePipeline, TranscriptionInfo
from faster_whisper.audio import decode_audio # Used in the helper

# Assuming these types are defined in api_types.py
from api_types import (
    DEFAULT_TIMESTAMP_GRANULARITIES,
    VALID_TIMESTAMP_GRANULARITY_SETS,
    CreateTranscriptionResponseJson,
    CreateTranscriptionResponseVerboseJson,
    TimestampGranularities, # Keep if used conceptually or elsewhere
    TranscriptionSegment,
)
# Assuming these are defined in dependencies.py
from dependencies import ConfigDependency, ModelManagerDependency
# Assuming this is defined in model_aliases.py
from model_aliases import ModelId
# Assuming these helpers exist in text_utils.py and expect (segment, index)
from text_utils import segments_to_srt, segments_to_text, segments_to_vtt

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- Router Setup ---
router = APIRouter(tags=["automatic-speech-recognition"])

# --- Type Definitions ---
type ResponseFormat = Literal["text", "json", "verbose_json", "srt", "vtt"]
DEFAULT_RESPONSE_FORMAT: ResponseFormat = "json"

# --- Helper function for audio decoding ---
async def decode_audio_from_upload(file: UploadFile) -> NDArray[np.float32]:
    """
    Decodes the uploaded audio file into a numpy array of float32 samples.
    Handles various audio formats using ffmpeg via the 'av' library.
    Raises HTTPException on decoding errors.
    """
    try:
        # Use faster_whisper's robust decoder which leverages 'av'
        audio_data = decode_audio(file.file)
        # Ensure float32 dtype as expected by whisper models
        if audio_data.dtype != np.float32:
             logger.debug(f"Casting audio data from {audio_data.dtype} to float32")
             audio_data = audio_data.astype(np.float32)
        return audio_data
    except av.error.InvalidDataError as e:
        logger.error(f"Failed to decode audio '{file.filename}'. Invalid data or unsupported format.")
        raise HTTPException(
            status_code=415, # Unsupported Media Type
            detail=f"Failed to decode audio: Unsupported file format or invalid data in '{file.filename}'.",
        ) from e
    except av.error.ValueError as e:
        # This specific ValueError from av often indicates an empty or corrupted file
        logger.error(f"Failed to decode audio '{file.filename}'. Value error (likely empty file or header issue).")
        raise HTTPException(
            status_code=400, # Bad Request
            detail=f"Failed to decode audio: The provided file '{file.filename}' is likely empty or corrupted.",
        ) from e
    except Exception as e:
        # Catch any other unexpected errors during decoding
        logger.exception(f"An unexpected error occurred while decoding audio '{file.filename}'.")
        raise HTTPException(status_code=500, detail="Internal server error during audio decoding.") from e
    finally:
        # Ensure the file stream is closed regardless of success or failure
        await file.close()


# --- Formatting function for non-streaming responses ---
def segments_to_response(
    segments: List[TranscriptionSegment], # Expect a List now
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> Response:
    """
    Formats the list of transcription segments into the specified non-streaming response format.
    """
    match response_format:
        case "text":
            # Assuming segments_to_text handles a list correctly
            return Response(segments_to_text(segments), media_type="text/plain")
        case "json":
            return Response(
                CreateTranscriptionResponseJson.from_segments(segments).model_dump_json(indent=2), # Add indent for readability
                media_type="application/json",
            )
        case "verbose_json":
            return Response(
                CreateTranscriptionResponseVerboseJson.from_segments(segments, transcription_info).model_dump_json(indent=2), # Add indent
                media_type="application/json",
            )
        case "vtt":
            # Iterate and call helper per segment, adding the VTT header
            vtt_output = "WEBVTT\n\n" + "".join(
                segments_to_vtt(segment, i) for i, segment in enumerate(segments)
            )
            return Response(vtt_output, media_type="text/vtt")
        case "srt":
             # Iterate and call helper per segment
             srt_output = "".join(
                 segments_to_srt(segment, i) for i, segment in enumerate(segments)
             )
             # Common media type for SRT files
             return Response(srt_output, media_type="application/x-subrip")
        # This case should ideally not be reached if ResponseFormat is used correctly
        case _:
             logger.error(f"Invalid response_format '{response_format}' encountered in segments_to_response.")
             # Fallback to plain text or raise an internal error
             return Response(segments_to_text(segments), media_type="text/plain")


# --- SSE formatting helper ---
def format_as_sse(data: str) -> str:
    """Formats data string into Server-Sent Event format."""
    lines = data.strip().split('\n')
    return "".join(f"data: {line}\n" for line in lines) + "\n"


# --- Formatting function for streaming responses ---
def segments_to_streaming_response(
    segments: List[TranscriptionSegment], # Expect a List now
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> StreamingResponse:
    """
    Creates a StreamingResponse that sends transcription segments one by one
    in the specified format using Server-Sent Events (SSE).
    """
    async def segment_responses() -> AsyncGenerator[str, None]:
        """Async generator yielding formatted segments for SSE."""
        # VTT needs a header sent first
        if response_format == "vtt":
            yield format_as_sse("WEBVTT\n") # Send header once

        # Iterate through the list of segments
        for segment_index, segment in enumerate(segments):
            data_to_send = ""
            try:
                if response_format == "text":
                    data_to_send = segment.text
                elif response_format == "json":
                    # Stream each segment as a standalone JSON object line
                    data_to_send = CreateTranscriptionResponseJson(text=segment.text).model_dump_json()
                elif response_format == "verbose_json":
                    # Stream each segment as a verbose JSON object line
                    data_to_send = CreateTranscriptionResponseVerboseJson(
                        task=transcription_info.transcription_options.get("task", "transcribe"),
                        language=transcription_info.language,
                        duration=transcription_info.duration, # Overall duration in each event
                        text=segment.text,
                        words=getattr(segment, 'words', []), # Safely get words
                        segments=[segment] # Send full segment data
                    ).model_dump_json()
                elif response_format == "vtt":
                     # Call helper with segment and index
                     data_to_send = segments_to_vtt(segment, segment_index)
                elif response_format == "srt":
                    # Call helper with segment and index
                    data_to_send = segments_to_srt(segment, segment_index)
                else:
                     logger.warning(f"Unsupported streaming format encountered: {response_format}")
                     continue # Skip sending data for this segment

                if data_to_send:
                    yield format_as_sse(data_to_send)
                    await asyncio.sleep(0.001) # Small sleep prevents overwhelming the connection

            except Exception as e:
                logger.error(f"Error formatting segment {segment_index} for streaming ({response_format}): {e}")
                # Optionally yield an error message event
                # yield format_as_sse(f'{{"error": "Error processing segment {segment_index}"}}')

        # Optional: Send a final marker if client expects it
        # yield format_as_sse("[DONE]")

    # Return the StreamingResponse using the async generator
    return StreamingResponse(segment_responses(), media_type="text/event-stream")


# --- Helper for timestamp_granularities[] parameter ---
async def get_timestamp_granularities(request: Request) -> List[Literal["segment", "word"]]:
    """
    Parses and validates the 'timestamp_granularities[]' form field.
    Returns a list of valid requested granularities or the default.
    Raises HTTPException on invalid input.
    """
    form = await request.form()
    # Use getlist to handle multiple values for the same key (e.g., ...granularities[]=word&...granularities[]=segment)
    raw_values = form.getlist("timestamp_granularities[]")

    # If the key wasn't provided at all or was empty list, return default
    if not raw_values:
        return DEFAULT_TIMESTAMP_GRANULARITIES

    # Check if the provided values are valid options
    provided_set = set(raw_values)
    valid_options = {"segment", "word"}
    invalid_values = provided_set - valid_options

    if invalid_values:
         raise HTTPException(
            status_code=422, # Unprocessable Entity
            detail=(f"Invalid value(s) provided for 'timestamp_granularities[]': {list(invalid_values)}. "
                    f"Allowed values are 'segment' and 'word'.")
         )

    # Return the validated list of unique values provided
    # Type hint ensures we return the expected literals
    result_list: List[Literal["segment", "word"]] = sorted(list(provided_set & valid_options)) # Sort for consistency
    return result_list


# --- API Endpoints ---

@router.post(
    "/v1/audio/translations",
    summary="Translate Audio to English",
    description="Translates audio into English text using the specified model.",
    response_model=None, # Response handled manually based on format/streaming
)
async def translate_file(
    # Dependencies first for cleaner signature
    config: ConfigDependency,
    model_manager: ModelManagerDependency,
    # Required parameters
    file: Annotated[UploadFile, File(description="The audio file object (e.g., wav, mp3, m4a, ogg) to translate.")],
    model: Annotated[ModelId, Form(description="ID of the model to use for translation (must support translation task).", example="Systran/faster-distil-whisper-large-v3")],
    # Optional parameters with defaults
    prompt: Annotated[str | None, Form(description="An optional text to guide the model's style or continue a previous audio segment.")] = None,
    response_format: Annotated[ResponseFormat, Form(description="The desired format for the translation output.")] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form(description="The sampling temperature (0-1). Higher values make output more random.", ge=0.0, le=1.0)] = 0.0,
    stream: Annotated[bool, Form(description="Whether to stream back partial progress using Server-Sent Events.")] = False,
    vad_filter: Annotated[bool, Form(description="Enable Voice Activity Detection (VAD) filter to potentially remove silence.")] = False,
) -> Response | StreamingResponse:
    """
    Handles audio file uploads, performs translation, and returns the result
    in the specified format, either fully or streamed.
    """
    logger.info(f"Received translation request for file '{file.filename}', model '{model}', format '{response_format}', stream={stream}")
    audio_data = await decode_audio_from_upload(file) # Decode using helper

    # Use context manager for model loading/unloading
    with model_manager.load_model(model) as whisper:
        # Check config for batched mode (if applicable and implemented)
        whisper_pipeline = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
        try:
            segments_generator, transcription_info = whisper_pipeline.transcribe(
                audio=audio_data, # Pass decoded numpy array
                task="translate", # Specify translation task
                initial_prompt=prompt,
                temperature=temperature,
                vad_filter=vad_filter,
                word_timestamps=False, # Word timestamps are not applicable to translation task in Whisper
            )

            # Convert faster-whisper segments to Pydantic models (works for generators)
            # IMPORTANT: Consume the generator here into a list
            pydantic_segments = TranscriptionSegment.from_faster_whisper_segments(segments_generator)
            segments_list = list(pydantic_segments)

            logger.info(f"Translation completed for '{file.filename}': Language={transcription_info.language}, Duration={transcription_info.duration:.2f}s, Segments={len(segments_list)}")

            if stream:
                logger.debug(f"Streaming translation response (format: {response_format})")
                # Pass the generated list to the streaming function
                return segments_to_streaming_response(segments_list, transcription_info, response_format)
            else:
                logger.debug(f"Returning full translation response (format: {response_format})")
                # Pass the generated list to the non-streaming function
                return segments_to_response(segments_list, transcription_info, response_format)

        except Exception as e:
            # Log the full traceback for internal errors
            logger.exception(f"Error during translation for {file.filename} with model {model}")
            raise HTTPException(status_code=500, detail=f"Translation failed due to an internal server error: {str(e)}")


@router.post(
    "/v1/audio/transcriptions",
    summary="Transcribe Audio",
    description="Transcribes audio into the input language text using the specified model.",
    response_model=None, # Response handled manually
)
async def transcribe_file(
    # Dependencies first
    request: Request, # Needed early for get_timestamp_granularities via Depends
    config: ConfigDependency,
    model_manager: ModelManagerDependency,
    # Required parameters
    file: Annotated[UploadFile, File(description="The audio file object (e.g., wav, mp3, m4a, ogg) to transcribe.")],
    model: Annotated[ModelId, Form(description="ID of the model to use for transcription.", example="Systran/faster-whisper-large-v3")],
    # Optional parameters with defaults
    language: Annotated[str | None, Form(description="The language of the input audio (ISO 639-1 or BCP-47 format). If None, language is auto-detected.", example="en")] = None,
    prompt: Annotated[str | None, Form(description="An optional text to guide the model's style or continue a previous audio segment.")] = None,
    response_format: Annotated[ResponseFormat, Form(description="The desired format for the transcription output.")] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form(description="The sampling temperature (0-1). Higher values make output more random.", ge=0.0, le=1.0)] = 0.0,
    # Use Depends for the helper function to handle 'timestamp_granularities[]'
    timestamp_granularities: List[Literal["segment", "word"]] = Depends(get_timestamp_granularities),
    stream: Annotated[bool, Form(description="Whether to stream back partial progress using Server-Sent Events.")] = False,
    hotwords: Annotated[str | None, Form(description="A string of 'hotwords' or specific terms to provide context, improving recognition of these words (syntax may depend on model).")] = None,
    vad_filter: Annotated[bool, Form(description="Enable Voice Activity Detection (VAD) filter to potentially remove silence.")] = False,
    # beam_size: Annotated[int, Form(description="Beam size for decoding.")] = 5, # Example if you want to expose beam_size
) -> Response | StreamingResponse:
    """
    Handles audio file uploads, performs transcription, and returns the result
    in the specified format, either fully or streamed. Supports language detection,
    timestamps, and other Whisper features.
    """

    word_timestamps_requested = "word" in timestamp_granularities

    # Validate timestamp request vs response format compatibility
    if word_timestamps_requested and response_format != "verbose_json":
         logger.warning(
            f"Requested word timestamps ('timestamp_granularities[]' included 'word') but response_format is '{response_format}'. "
            f"Word timestamps are only available in 'verbose_json'. Ignoring request for word timestamps."
         )
         word_timestamps_requested = False # Override if format doesn't support it

    # Log if segment timestamps requested but format doesn't explicitly show them (like 'text' or 'json')
    if "segment" in timestamp_granularities and response_format not in ["verbose_json", "vtt", "srt"]:
         logger.debug(
              f"Requested segment timestamps ('timestamp_granularities[]' included 'segment') but response_format is '{response_format}'. "
              f"Timestamps are only explicitly shown in 'verbose_json', 'vtt', or 'srt'."
         )

    logger.info(f"Received transcription request for file '{file.filename}', model '{model}', language '{language}', format '{response_format}', stream={stream}, timestamps={timestamp_granularities}")
    audio_data = await decode_audio_from_upload(file) # Decode using helper

    with model_manager.load_model(model) as whisper:
        whisper_pipeline = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
        try:
            # Prepare arguments for transcribe method
            transcribe_args = {
                "audio": audio_data,
                "task": "transcribe",
                "language": language,
                "initial_prompt": prompt,
                "word_timestamps": word_timestamps_requested, # Pass based on validated request
                "temperature": temperature,
                "vad_filter": vad_filter,
                "beam_size": 5, # Set beam_size if desired
                # Add hotwords only if provided and not empty
                **({"hotwords": hotwords} if hotwords else {}),
            }

            segments_generator, transcription_info = whisper_pipeline.transcribe(**transcribe_args)

            # Log info after transcription starts (info object is available immediately)
            logger.info(f"Transcription info for '{file.filename}': Detected Language={transcription_info.language}, Confidence={transcription_info.language_probability:.2f}, Duration={transcription_info.duration:.2f}s")

            # Convert faster-whisper segments to Pydantic models
            # IMPORTANT: Consume the generator here into a list
            pydantic_segments = TranscriptionSegment.from_faster_whisper_segments(segments_generator)
            segments_list = list(pydantic_segments)

            logger.info(f"Transcription completed for file '{file.filename}' with {len(segments_list)} segments.")

            if stream:
                logger.debug(f"Streaming transcription response (format: {response_format})")
                # Pass the generated list to the streaming function
                return segments_to_streaming_response(segments_list, transcription_info, response_format)
            else:
                logger.debug(f"Returning full transcription response (format: {response_format})")
                # Pass the generated list to the non-streaming function
                return segments_to_response(segments_list, transcription_info, response_format)

        except Exception as e:
            # Log the full traceback for internal errors
            logger.exception(f"Error during transcription for {file.filename} with model {model}")
            # Return a 500 error to the client
            raise HTTPException(status_code=500, detail=f"Transcription failed due to an internal server error: {str(e)}")