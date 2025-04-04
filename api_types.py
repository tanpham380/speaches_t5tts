#api_types.py
from __future__ import annotations # Ensure forward references work smoothly

from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type, Union # Added Type, Union, Dict, Any

import faster_whisper.transcribe
from pydantic import BaseModel, Field, computed_field, field_validator

from .text_utils import segments_to_text

DEFAULT_F5_TTS_MODEL_ID = "SWivid/F5-TTS_v1_Base"
DEFAULT_E2_TTS_MODEL_ID = "SWivid/E2-TTS_Base"

type ResponseFormat = Literal["mp3", "flac", "wav", "pcm"]
SUPPORTED_RESPONSE_FORMATS = ("mp3", "flac", "wav", "pcm")
DEFAULT_RESPONSE_FORMAT = "mp3"



# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10909
class TranscriptionWord(BaseModel):
    start: float
    end: float
    word: str
    probability: float

    @classmethod
    def from_segments(cls, segments: Iterable["TranscriptionSegment"]) -> list["TranscriptionWord"]:
        words: list[TranscriptionWord] = []
        for segment in segments:
            # NOTE: a temporary "fix" for https://github.com/speaches-ai/speaches/issues/58.
            # TODO: properly address the issue
            assert segment.words is not None, (
                "Segment must have words. If you are using an API ensure `timestamp_granularities[]=word` is set"
            )
            words.extend(segment.words)
        return words

    def offset(self, seconds: float) -> None:
        self.start += seconds
        self.end += seconds
        
class F5TTSModel(BaseModel):
    """
    Represents an available F5-TTS or E2-TTS model, suitable for listing via an API.
    Specific loading details (checkpoint, vocab, config) are typically associated
    with the `id` externally (e.g., in a configuration mapping or model manager).
    """
    id: str = Field(..., description="The unique identifier for the model (e.g., HF repo ID or a custom name).")
    created: int = Field(0, description="Approximate creation timestamp (Unix epoch seconds), often from model file metadata or 0 if unknown.")
    object: Literal["model"] = Field("model", description="Object type identifier.")
    owned_by: str = Field(..., description="The user or organization owning the model (e.g., 'SWivid', 'local').")
    language: list[str] | None = Field(default=["en", "zh"], description="List of supported language codes (ISO 639-1 or similar).")
    task: ModelTask = Field("text-to-speech", description="The primary task supported by the model.")
    architecture: Literal["DiT", "UNetT", "Unknown"] = Field("Unknown", description="The underlying model architecture (DiT for F5-TTS, UNetT for E2-TTS).")

    @field_validator('owned_by', mode='before')
    @classmethod
    def set_owned_by_from_id(cls, v: str | None, values: Any) -> str:
        """Automatically sets owned_by from the id if not provided and id is HF-like."""
        if v:
            return v
        model_id = values.data.get('id')
        if model_id and '/' in model_id:
            return model_id.split('/')[0]
        return "unknown" # Or raise error if required

# --- Generation Parameters ---
class F5TTSGenerationParams(BaseModel):
    """Parameters controlling the F5-TTS audio generation process."""
    speed: float = Field(1.0, description="Speed adjustment factor.", ge=0.1, le=3.0)
    nfe_step: int = Field(32, description="Number of Function Evaluations (denoising steps).", ge=4, le=128)
    cross_fade_duration: float = Field(0.15, description="Duration in seconds for cross-fading between internal chunks.", ge=0.0, le=1.0)
    remove_silence: bool = Field(False, description="Attempt to remove leading/trailing silences from the output.")
    output_sample_rate: int | None = Field(None, description="If set, resample the output audio to this sample rate (Hz).")


# --- Generation Request ---

class F5TTSGenerationRequest(BaseModel):
    """Represents the input needed to generate audio using F5-TTS."""
    ref_audio_path: str = Field(..., description="Path or URI (e.g., file://, http://, hf://) to the reference audio file.")
    gen_text: str = Field(..., description="The text to be synthesized.")
    ref_text: str | None = Field(None, description="Optional transcript of the reference audio. If None, transcription may be attempted.")
    parameters: F5TTSGenerationParams = Field(default_factory=F5TTSGenerationParams, description="Synthesis control parameters.")
    # Note: The specific F5-TTS model instance (or its identifier to load it) is typically
    # handled by the context (e.g., the API endpoint or calling function) rather than
    # being part of the request payload itself, unless the API is designed to select
    # the model based on the request.

# --- Generation Response Metadata (Optional) ---

class F5TTSGenerationResponseMetadata(BaseModel):
    """Optional metadata accompanying the generated audio stream or sent afterwards."""
    model_id_used: str = Field(..., description="Identifier of the specific F5-TTS model checkpoint/config used for generation.")
    input_characters: int = Field(..., description="Number of characters in the input `gen_text`.", ge=0)
    reference_audio_path: str = Field(..., description="The reference audio path provided in the request.")
    effective_reference_text: str | None = Field(None, description="The reference text used (either provided or transcribed).")
    output_duration_seconds: float = Field(..., description="Duration of the generated audio in seconds.", ge=0.0)
    output_sample_rate: int = Field(..., description="Actual sample rate of the generated audio data (Hz).")
    generation_time_seconds: float = Field(..., description="Time taken for the core audio generation process in seconds.", ge=0.0)
    parameters_used: F5TTSGenerationParams = Field(..., description="The generation parameters that were actually used.")
    request_id: str | None = Field(None, description="Optional unique identifier for tracking the request.")


class ModifiedSpeechRequestParams(BaseModel):
    """Helper model for form parameters (not used directly in route signature)"""
    model: str = Field(default=DEFAULT_F5_TTS_MODEL_ID)
    input: str
    response_format: ResponseFormat = DEFAULT_RESPONSE_FORMAT
    speed: float = 1.0
    sample_rate: int | None = None
    reference_text: str | None = None # Optional ref text
    # Add other parameters from F5TTSGenerationParams if needed as Form fields
    nfe_step: int = 32
    cross_fade_duration: float = 0.15
    remove_silence: bool = False

# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10938
class TranscriptionSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[TranscriptionWord] | None

    @classmethod
    def from_faster_whisper_segments(
        cls, segments: Iterable[faster_whisper.transcribe.Segment]
    ) -> Iterable["TranscriptionSegment"]:
        for segment in segments:
            yield cls(
                id=segment.id,
                seek=segment.seek,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                tokens=segment.tokens,
                temperature=segment.temperature or 0,  # FIX: hardcoded
                avg_logprob=segment.avg_logprob,
                compression_ratio=segment.compression_ratio,
                no_speech_prob=segment.no_speech_prob,
                words=[
                    TranscriptionWord(
                        start=word.start,
                        end=word.end,
                        word=word.word,
                        probability=word.probability,
                    )
                    for word in segment.words
                ]
                if segment.words is not None
                else None,
            )


# https://platform.openai.com/docs/api-reference/audio/json-object
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10924
class CreateTranscriptionResponseJson(BaseModel):
    text: str

    @classmethod
    def from_segments(cls, segments: list[TranscriptionSegment]) -> "CreateTranscriptionResponseJson":
        return cls(text=segments_to_text(segments))


# https://platform.openai.com/docs/api-reference/audio/verbose-json-object
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L11007
class CreateTranscriptionResponseVerboseJson(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: list[TranscriptionWord] | None
    segments: list[TranscriptionSegment]

    @classmethod
    def from_segment(
        cls, segment: TranscriptionSegment, transcription_info: faster_whisper.transcribe.TranscriptionInfo
    ) -> "CreateTranscriptionResponseVerboseJson":
        return cls(
            language=transcription_info.language,
            duration=segment.end - segment.start,
            text=segment.text,
            words=segment.words if transcription_info.transcription_options.word_timestamps else None,
            segments=[segment],
        )

    @classmethod
    def from_segments(
        cls, segments: list[TranscriptionSegment], transcription_info: faster_whisper.transcribe.TranscriptionInfo
    ) -> "CreateTranscriptionResponseVerboseJson":
        return cls(
            language=transcription_info.language,
            duration=transcription_info.duration,
            text=segments_to_text(segments),
            segments=segments,
            words=TranscriptionWord.from_segments(segments)
            if transcription_info.transcription_options.word_timestamps
            else None,
        )


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L8730
class ListModelsResponse(BaseModel):
    data: list["Model"]
    object: Literal["list"] = "list"


ModelTask = Literal["automatic-speech-recognition", "text-to-speech"]  # TODO: add "voice-activity-detection"


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L11146
class Model(BaseModel):
    id: str
    """The model identifier, which can be referenced in the API endpoints."""
    created: int = 0
    """The Unix timestamp (in seconds) when the model was created."""
    object: Literal["model"] = "model"
    """The object type, which is always "model"."""
    owned_by: str
    """The organization that owns the model."""
    language: list[str] | None = None
    """List of ISO 639-3 supported by the model. It's possible that the list will be empty. This field is not a part of the OpenAI API spec and is added for convenience."""

    task: ModelTask  # TODO: make a list?


# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L10909
TimestampGranularities = list[Literal["segment", "word"]]


DEFAULT_TIMESTAMP_GRANULARITIES: TimestampGranularities = ["segment"]
TIMESTAMP_GRANULARITIES_COMBINATIONS: list[TimestampGranularities] = [
    [],  # should be treated as ["segment"]. https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities
    ["segment"],
    ["word"],
    ["word", "segment"],
    ["segment", "word"],  # same as ["word", "segment"] but order is different
]




class Voice(BaseModel):
    """Represents an available voice identity, potentially across different TTS engines."""
    id: str = Field(..., description="The unique identifier for the voice (e.g., 'alloy', 'my_custom_voice_1').")
    name: Optional[str] = Field(None, description="A user-friendly name for the voice (might be same as id).")
    # You could add engine-specific details here using conditional fields or a union if needed,
    # but for a simple listing, ID and maybe name are often sufficient.
    # Example: engine: Literal["openai", "f5tts-custom", "piper"] = "unknown"
    # Example: associated_model: Optional[str] = None # Which model ID(s) this voice works with

    # If you want the response to exactly match the old Piper structure:
    # model_id: str | None = None # The model this voice belongs to (e.g., F5-TTS_v1)
    # voice_id: str # The specific voice ID within that model (e.g., my_custom_voice_1)
    # created: int = 0 # Timestamp might be hard to get consistently
    # owned_by: str | None = None
    # sample_rate: int | None = None # Sample rate of the *reference* audio maybe?
    # object: Literal["voice"] = "voice"

    # @computed_field(examples=["F5-TTS_v1/my_custom_voice_1"])
    # @cached_property
    # def combined_id(self) -> str | None:
    #      if self.model_id and self.voice_id:
    #           return f"{self.model_id}/{self.voice_id}"
    #      return self.voice_id # Fallback