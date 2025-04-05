# routers/stt.py

import asyncio
from collections.abc import Generator, Iterable
import logging
from typing import Annotated, Literal, List
import av

# --- ADD THESE IMPORTS ---
import numpy as np
from numpy.typing import NDArray
# --- END IMPORTS ---

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
from faster_whisper.audio import decode_audio

from api_types import (
    DEFAULT_TIMESTAMP_GRANULARITIES,
    VALID_TIMESTAMP_GRANULARITY_SETS,
    CreateTranscriptionResponseJson,
    CreateTranscriptionResponseVerboseJson,
    TimestampGranularities,
    TranscriptionSegment,
)
from dependencies import ConfigDependency, ModelManagerDependency # Keep these type aliases
from model_aliases import ModelId
from text_utils import segments_to_srt, segments_to_text, segments_to_vtt

logger = logging.getLogger(__name__)

router = APIRouter(tags=["automatic-speech-recognition"])

type ResponseFormat = Literal["text", "json", "verbose_json", "srt", "vtt"]
DEFAULT_RESPONSE_FORMAT: ResponseFormat = "json"

# --- Helper function for audio decoding (updated type hint) ---
# Use the imported types for the annotation
async def decode_audio_from_upload(file: UploadFile) -> NDArray[np.float32]:
    try:
        audio_data = decode_audio(file.file)
        # Ensure the return type is float32 if decode_audio doesn't guarantee it
        # (faster_whisper's decode_audio usually returns float32)
        if audio_data.dtype != np.float32:
             logger.debug(f"Casting audio data from {audio_data.dtype} to float32")
             audio_data = audio_data.astype(np.float32)
        return audio_data
    except av.error.InvalidDataError as e:
        logger.error(f"Failed to decode audio '{file.filename}'. Invalid data or unsupported format.")
        raise HTTPException(
            status_code=415,
            detail=f"Failed to decode audio: Unsupported file format or invalid data in '{file.filename}'.",
        ) from e
    except av.error.ValueError as e:
        logger.error(f"Failed to decode audio '{file.filename}'. Value error (likely empty file).")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode audio: The provided file '{file.filename}' is likely empty or corrupted.",
        ) from e
    except Exception as e:
        logger.exception(f"An unexpected error occurred while decoding audio '{file.filename}'.")
        raise HTTPException(status_code=500, detail="Internal server error during audio decoding.") from e
    finally:
        await file.close()

# --- Functions segments_to_response, format_as_sse, segments_to_streaming_response remain the same ---
def segments_to_response(
    segments: Iterable[TranscriptionSegment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> Response:
    segments = list(segments)
    match response_format:
        case "text":
            return Response(segments_to_text(segments), media_type="text/plain")
        case "json":
            return Response(
                CreateTranscriptionResponseJson.from_segments(segments).model_dump_json(),
                media_type="application/json",
            )
        case "verbose_json":
            return Response(
                CreateTranscriptionResponseVerboseJson.from_segments(segments, transcription_info).model_dump_json(),
                media_type="application/json",
            )
        case "vtt":
            return Response(
                "".join(segments_to_vtt(segment, i) for i, segment in enumerate(segments)), media_type="text/vtt"
            )
        case "srt":
            return Response(
                "".join(segments_to_srt(segment, i) for i, segment in enumerate(segments)), media_type="text/plain"
            )


def format_as_sse(data: str) -> str:
    return f"data: {data}\n\n"


def segments_to_streaming_response(
    segments: Iterable[TranscriptionSegment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> StreamingResponse:
    def segment_responses() -> Generator[str, None, None]:
        segments_list = list(segments) # Consume iterator once if needed multiple times
        # Ensure transcription_info is accessible if needed inside loop
        for i, segment in enumerate(segments_list):
            if response_format == "text":
                data = segment.text
            elif response_format == "json":
                # For streaming json, send each segment as a complete JSON object
                data = CreateTranscriptionResponseJson(text=segment.text).model_dump_json()
            elif response_format == "verbose_json":
                # For streaming verbose_json, send each segment as a verbose object
                # Need language/duration from info, but text/words/segments from the *current* segment
                data = CreateTranscriptionResponseVerboseJson(
                    task=transcription_info.transcription_options.get("task", "transcribe"), # Get task from info if possible
                    language=transcription_info.language,
                    duration=transcription_info.duration, # Use overall duration? Or segment duration? Check OpenAI spec for streaming. Using segment end for now.
                    text=segment.text,
                    words=segment.words,
                    segments=[segment] # Stream one segment at a time
                ).model_dump_json()
            elif response_format == "vtt":
                 # VTT header needs to be sent first if streaming line-by-line
                 # This simple approach sends full blocks per segment
                 if i == 0:
                     yield format_as_sse(f"WEBVTT\n\n{segments_to_vtt(segment, i)}")
                 else:
                      yield format_as_sse(segments_to_vtt(segment, i))
                 continue # Skip default format_as_sse for VTT after handling
            elif response_format == "srt":
                data = segments_to_srt(segment, i)
            else:
                 # Fallback or error for unknown format
                 logger.warning(f"Unsupported streaming format: {response_format}")
                 continue

            if response_format != "vtt": # Already handled VTT yield
                 yield format_as_sse(data)

        # Optionally send a final message or close marker if needed by SSE spec/client
        # yield format_as_sse("[DONE]") # Example

    return StreamingResponse(segment_responses(), media_type="text/event-stream")


# --- Updated Endpoints ---

@router.post(
    "/v1/audio/translations",
    response_model=None,
)
async def translate_file(
    # --- MOVE DEPENDENCIES BEFORE DEFAULT ARGUMENTS ---
    config: ConfigDependency,
    model_manager: ModelManagerDependency,
    # --- END MOVE ---
    # Non-default File parameter
    file: Annotated[UploadFile, File(description="The audio file object (wav, mp3, m4a, etc.) to translate.")],
    # Non-default Form parameter
    model: Annotated[ModelId, Form(description="ID of the model to use for translation." , example="erax-ai/EraX-WoW-Turbo-V1.1-CT2") ],
    # Default Form parameters
    prompt: Annotated[str | None, Form(description="An optional text to guide the model's style or continue a previous audio segment.")] = None,
    response_format: Annotated[ResponseFormat, Form(description="The format of the transcript output.")] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form(description="The sampling temperature, between 0 and 1.")] = 0.0,
    stream: Annotated[bool, Form(description="Whether to stream back partial progress.")] = False,
    vad_filter: Annotated[bool, Form(description="Enable VAD filter")] = False,
) -> Response | StreamingResponse:

    logger.info(f"Received translation request for file '{file.filename}', model '{model}', format '{response_format}'")
    audio_data = await decode_audio_from_upload(file) # Decode the audio here

    with model_manager.load_model(model) as whisper:
        whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
        segments_generator, transcription_info = whisper_model.transcribe(
            audio_data, # Pass the decoded numpy array
            task="translate",
            initial_prompt=prompt,
            temperature=temperature,
            vad_filter=vad_filter,
            word_timestamps=False,
        )
        pydantic_segments = TranscriptionSegment.from_faster_whisper_segments(segments_generator)

        if stream:
            logger.debug(f"Streaming translation response (format: {response_format})")
            return segments_to_streaming_response(pydantic_segments, transcription_info, response_format)
        else:
            logger.debug(f"Returning full translation response (format: {response_format})")
            segments_list = list(pydantic_segments)
            return segments_to_response(segments_list, transcription_info, response_format)


# Keep this helper for timestamp_granularities[]
async def get_timestamp_granularities(request: Request) -> List[Literal["segment", "word"]]:
    form = await request.form()
    raw_values = form.getlist("timestamp_granularities[]")
    if not raw_values:
        return DEFAULT_TIMESTAMP_GRANULARITIES

    provided_set = set(raw_values)
    if provided_set not in VALID_TIMESTAMP_GRANULARITY_SETS:
        raise HTTPException(
            status_code=422,
            detail=(f"Invalid value combination for 'timestamp_granularities[]': {raw_values}. "
                    f"Allowed combinations (any order): [], ['segment'], ['word'], ['segment', 'word'].")
        )

    if not provided_set:
         return DEFAULT_TIMESTAMP_GRANULARITIES
    else:
         validated_list = list(provided_set)
         result_list: List[Literal["segment", "word"]] = [item for item in validated_list if item in ("segment", "word")]
         return result_list


@router.post(
    "/v1/audio/transcriptions",
    response_model=None,
)
async def transcribe_file(
    # --- MOVE DEPENDENCIES BEFORE DEFAULT ARGUMENTS ---
    request: Request, # Keep request early for timestamp helper
    config: ConfigDependency,
    model_manager: ModelManagerDependency,
    # --- END MOVE ---
    # Non-default File parameter
    file: Annotated[UploadFile, File(description="The audio file object (wav, mp3, m4a, etc.) to transcribe.")],
    # Non-default Form parameter
    model: Annotated[ModelId, Form(description="ID of the model to use for transcription." , example="erax-ai/EraX-WoW-Turbo-V1.1-CT2")],
    # Default Form parameters
    language: Annotated[str | None, Form(description="The language of the input audio." , example="vi")] = None,
    prompt: Annotated[str | None, Form(description="An optional text to guide the model's style or continue a previous audio segment." , example="") ] = None,
    response_format: Annotated[ResponseFormat, Form(description="The format of the transcript output.")] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form(description="The sampling temperature, between 0 and 1.")] = 0.0,
    stream: Annotated[bool, Form(description="Whether to stream back partial progress.")] = False,
    hotwords: Annotated[str | None, Form(description="A list of words to help the model recognize faster.")] = None,
    vad_filter: Annotated[bool, Form(description="Enable VAD filter")] = False,
    # timestamp_granularities handled by the helper function below
) -> Response | StreamingResponse:

    timestamp_granularities = await get_timestamp_granularities(request)
    word_timestamps_requested = "word" in timestamp_granularities

    if word_timestamps_requested and response_format != "verbose_json":
         logger.warning(
            f"Requested word timestamps ('timestamp_granularities[]' included 'word') but response_format is '{response_format}'. Word timestamps only available in 'verbose_json'. Ignoring."
         )
         word_timestamps_requested = False

    if "segment" in timestamp_granularities and response_format != "verbose_json":
         logger.debug(
              f"Requested segment timestamps ('timestamp_granularities[]' included 'segment' or was default) but response_format is '{response_format}'. Segment details only in 'verbose_json'."
         )


    logger.info(f"Received transcription request for file '{file.filename}', model '{model}', language '{language}', format '{response_format}'")
    audio_data = await decode_audio_from_upload(file)

    with model_manager.load_model(model) as whisper:
        whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
        segments_generator, transcription_info = whisper_model.transcribe(
            audio_data,
            task="transcribe",
            language=language,
            initial_prompt=prompt,
            word_timestamps=word_timestamps_requested,
            temperature=temperature,
            vad_filter=vad_filter,
            hotwords=hotwords,
        )
        pydantic_segments = TranscriptionSegment.from_faster_whisper_segments(segments_generator)

        if stream:
            logger.debug(f"Streaming transcription response (format: {response_format})")
            return segments_to_streaming_response(pydantic_segments, transcription_info, response_format)
        else:
            logger.debug(f"Returning full transcription response (format: {response_format})")
            segments_list = list(pydantic_segments)
            return segments_to_response(segments_list, transcription_info, response_format)