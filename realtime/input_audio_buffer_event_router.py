# realtime/input_audio_buffer_event_router.py
import base64
from io import BytesIO
import logging
from typing import Literal
# import soundfile as sf # Bỏ import
# from pathlib import Path # Bỏ import
import time
import numpy as np
import librosa # Đảm bảo đã cài: pip install librosa
from faster_whisper.transcribe import get_speech_timestamps
from faster_whisper.vad import VadOptions
from numpy.typing import NDArray
from openai.types.beta.realtime.error_event import Error
from pydantic import ValidationError

# Đảm bảo đường dẫn import này đúng
from realtime.context import SessionContext
from realtime.event_router import EventRouter
from realtime.input_audio_buffer import (
    MAX_VAD_WINDOW_SIZE_SAMPLES,
    MS_SAMPLE_RATE, # 16
    InputAudioBuffer,
    InputAudioBufferTranscriber,
    SAMPLE_RATE as INPUT_AUDIO_BUFFER_SAMPLE_RATE, # 16000
)
# Đảm bảo đường dẫn import này đúng
from mouble_types.realtime import (
    InputAudioBufferAppendEvent,
    InputAudioBufferClearedEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferCommittedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    TurnDetection,
    create_invalid_request_error,
)

MIN_AUDIO_BUFFER_DURATION_MS = 100
MAX_BUFFER_DURATION_MS = 5000 # Giữ lại giới hạn này

logger = logging.getLogger(__name__)

event_router = EventRouter()
empty_input_audio_buffer_commit_error = Error(
    type="invalid_request_error",
    message="Error committing input audio buffer: the buffer is empty.",
)

type SpeechTimestamp = dict[Literal["start", "end"], int]

# --- Các hàm tiện ích ---

# ***** THÊM LẠI HÀM RESAMPLE_AUDIO_DATA *****
def resample_audio_data(data: NDArray[np.float32], sample_rate: int, target_sample_rate: int) -> NDArray[np.float32]:
    """Resamples audio data using linear interpolation."""
    if sample_rate == target_sample_rate:
        return data
    ratio = target_sample_rate / sample_rate
    target_length = int(len(data) * ratio)
    if target_length == 0 and data.size > 0:
        target_length = 1 # Tránh lỗi chia cho 0 nếu ratio quá nhỏ
    elif target_length == 0 and data.size == 0:
        return np.array([], dtype=np.float32)
    resampled_data = np.interp(
        np.linspace(0, len(data), target_length, endpoint=False),
        np.arange(len(data)),
        data
    ).astype(np.float32)
    return resampled_data
# ********************************************

def to_ms_speech_timestamps(speech_timestamps: list[SpeechTimestamp]) -> list[SpeechTimestamp]:
    """Converts speech timestamps from samples (at 16kHz) to milliseconds."""
    if not speech_timestamps: return []
    converted_timestamps = []
    for ts in speech_timestamps:
        start_samples, end_samples = ts.get("start"), ts.get("end")
        if isinstance(start_samples, (int, np.integer)) and isinstance(end_samples, (int, np.integer)):
             start_ms = start_samples // MS_SAMPLE_RATE; end_ms = end_samples // MS_SAMPLE_RATE
             converted_timestamps.append({"start": start_ms, "end": end_ms})
        else: logger.warning(f"Skipping conversion for invalid timestamp sample format: {ts}")
    return converted_timestamps

# --- Logic VAD ---
def vad_detection_flow( input_audio_buffer: InputAudioBuffer, turn_detection: TurnDetection) -> InputAudioBufferSpeechStartedEvent | InputAudioBufferSpeechStoppedEvent | None:
    """Performs VAD on the latest audio window and returns start/stop events."""
    # logger.debug(f"VAD Check: threshold={turn_detection.threshold}, silence_duration_ms={turn_detection.silence_duration_ms}")
    if input_audio_buffer.data is None or input_audio_buffer.data.size == 0: return None
    vad_window_duration_ms = MAX_VAD_WINDOW_SIZE_SAMPLES // MS_SAMPLE_RATE
    audio_window = input_audio_buffer.data[-MAX_VAD_WINDOW_SIZE_SAMPLES:]
    min_required_samples = turn_detection.silence_duration_ms * MS_SAMPLE_RATE
    if audio_window.size < min_required_samples: return None
    try:
        raw_speech_timestamps_samples = list(get_speech_timestamps(audio_window, vad_options=VadOptions(threshold=turn_detection.threshold, min_silence_duration_ms=turn_detection.silence_duration_ms, speech_pad_ms=turn_detection.prefix_padding_ms), sampling_rate=INPUT_AUDIO_BUFFER_SAMPLE_RATE))
        speech_timestamps_ms = to_ms_speech_timestamps(raw_speech_timestamps_samples)
    except Exception as e: logger.error(f"Error during VAD processing: {e}", exc_info=True); return None
    if len(speech_timestamps_ms) > 1: logger.warning(f"VAD found multiple speech segments (ms): {speech_timestamps_ms}")
    last_speech_segment_ms = speech_timestamps_ms[-1] if speech_timestamps_ms else None
    now_ms = input_audio_buffer.duration_ms
    if input_audio_buffer.vad_state.audio_start_ms is None:
        if last_speech_segment_ms is not None:
            start_in_window_ms = last_speech_segment_ms["start"]; window_start_offset_ms = max(0, now_ms - vad_window_duration_ms); absolute_start_ms = window_start_offset_ms + start_in_window_ms
            input_audio_buffer.vad_state.audio_start_ms = max(0, absolute_start_ms); logger.info(f"VAD detected speech start at ~{input_audio_buffer.vad_state.audio_start_ms} ms (Buffer duration: {now_ms} ms)")
            return InputAudioBufferSpeechStartedEvent(item_id=input_audio_buffer.id, audio_start_ms=input_audio_buffer.vad_state.audio_start_ms)
        else: return None
    else:
        speech_stopped = False
        if last_speech_segment_ms is None: speech_stopped = True
        elif last_speech_segment_ms["end"] < (vad_window_duration_ms - turn_detection.silence_duration_ms): speech_stopped = True
        if speech_stopped:
            estimated_end_ms = 0
            if raw_speech_timestamps_samples:
                window_start_in_buffer_samples = max(0, input_audio_buffer.data.size - MAX_VAD_WINDOW_SIZE_SAMPLES); last_segment_end_samples = raw_speech_timestamps_samples[-1].get('end', 0)
                absolute_end_samples = window_start_in_buffer_samples + last_segment_end_samples; estimated_end_ms_from_vad = absolute_end_samples // MS_SAMPLE_RATE
                estimated_end_ms = max(input_audio_buffer.vad_state.audio_start_ms, estimated_end_ms_from_vad)
            else:
                estimated_end_ms_no_vad = max(input_audio_buffer.vad_state.audio_start_ms, now_ms - turn_detection.silence_duration_ms)
                estimated_end_ms = estimated_end_ms_no_vad
            estimated_end_ms = max(estimated_end_ms, input_audio_buffer.vad_state.audio_start_ms + MIN_AUDIO_BUFFER_DURATION_MS)
            estimated_end_ms = min(estimated_end_ms, now_ms - turn_detection.prefix_padding_ms)
            if estimated_end_ms > input_audio_buffer.vad_state.audio_start_ms:
                 input_audio_buffer.vad_state.audio_end_ms = estimated_end_ms; logger.info(f"VAD detected speech end at ~{input_audio_buffer.vad_state.audio_end_ms} ms (Buffer duration: {now_ms} ms)")
                 return InputAudioBufferSpeechStoppedEvent(item_id=input_audio_buffer.id, audio_end_ms=input_audio_buffer.vad_state.audio_end_ms)
            else: logger.warning(f"Calculated end_ms ({estimated_end_ms}) not valid. Not stopping yet."); return None
        else: return None

# --- Event Handlers ---

def _get_chunk_counter(ctx: SessionContext) -> int:
    """Gets and increments the chunk counter for the session."""
    if not hasattr(ctx, '_chunk_counter'): setattr(ctx, '_chunk_counter', 0)
    ctx._chunk_counter += 1
    return ctx._chunk_counter

@event_router.register("input_audio_buffer.append")
def handle_input_audio_buffer_append(ctx: SessionContext, event: InputAudioBufferAppendEvent) -> None:
    """Handles incoming audio chunks, reads, resamples, appends, checks max duration, runs VAD."""
    chunk_index = _get_chunk_counter(ctx)
    session_id = ctx.session.id if ctx.session else "unknown_session"
    INCOMING_SR = 24000; TARGET_SR = INPUT_AUDIO_BUFFER_SAMPLE_RATE

    try:
        audio_bytes = base64.b64decode(event.audio)
        if not audio_bytes: logger.warning(f"Chunk {chunk_index}: Empty audio after base64."); return
        try:
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_chunk_raw_24k = audio_int16.astype(np.float32) / 32768.0
            if audio_chunk_raw_24k.size == 0: logger.warning(f"Chunk {chunk_index}: Empty audio after frombuffer."); return
        except Exception as read_err: logger.error(f"Chunk {chunk_index}: Failed to read bytes: {read_err}", exc_info=True); return

        audio_chunk_16k = None
        use_librosa = True # Đặt thành False nếu muốn dùng hàm np.interp cũ

        if INCOMING_SR != TARGET_SR:
            if use_librosa:
                try: audio_chunk_16k = librosa.resample(y=audio_chunk_raw_24k, orig_sr=INCOMING_SR, target_sr=TARGET_SR, res_type='soxr_hq')
                except Exception as resample_err: logger.error(f"Chunk {chunk_index}: Failed to resample using librosa: {resample_err}", exc_info=True); return
            else: audio_chunk_16k = resample_audio_data(audio_chunk_raw_24k, INCOMING_SR, TARGET_SR) # Hàm cũ
        else: audio_chunk_16k = audio_chunk_raw_24k

        if audio_chunk_16k is None or audio_chunk_16k.size == 0: logger.warning(f"Chunk {chunk_index}: Audio data is empty after resampling attempt."); return

        if not ctx.input_audio_buffers:
            logger.warning("No active input buffer. Creating new."); new_buffer = InputAudioBuffer(ctx.pubsub); ctx.input_audio_buffers[new_buffer.id] = new_buffer; input_audio_buffer_id = new_buffer.id
        else: input_audio_buffer_id = next(reversed(ctx.input_audio_buffers))
        if input_audio_buffer_id not in ctx.input_audio_buffers:
            logger.error(f"Buffer {input_audio_buffer_id} missing. Creating new."); new_buffer = InputAudioBuffer(ctx.pubsub); ctx.input_audio_buffers[new_buffer.id] = new_buffer; input_audio_buffer_id = new_buffer.id
            input_audio_buffer = new_buffer
        else: input_audio_buffer = ctx.input_audio_buffers[input_audio_buffer_id]

        input_audio_buffer.append(audio_chunk_16k)
        # logger.debug(f"Chunk {chunk_index}: Appended. Buffer {input_audio_buffer_id} duration: {input_audio_buffer.duration_ms} ms")

        force_commit = False
        if input_audio_buffer.vad_state.audio_end_ms is None and input_audio_buffer.duration_ms > MAX_BUFFER_DURATION_MS:
            logger.warning(f"Buffer {input_audio_buffer_id} ({input_audio_buffer.duration_ms}ms) exceeded max duration {MAX_BUFFER_DURATION_MS}ms. Forcing commit.")
            input_audio_buffer.vad_state.audio_end_ms = input_audio_buffer.duration_ms # Gán end_ms
            force_commit = True

        if not force_commit and input_audio_buffer.vad_state.audio_end_ms is None and ctx.session.turn_detection is not None and ctx.session.turn_detection.type == "server_vad":
             vad_event = vad_detection_flow(input_audio_buffer, ctx.session.turn_detection)
             if vad_event is not None: ctx.pubsub.publish_nowait(vad_event)

        if force_commit:
            try:
                commit_event = InputAudioBufferCommitEvent(type="input_audio_buffer.commit", item_id=input_audio_buffer_id)
                ctx.pubsub.publish_nowait(commit_event)
            except Exception as pub_err: logger.error(f"Failed to publish forced commit event: {pub_err}")

    except Exception as e:
        logger.error(f"Chunk {chunk_index}: Unhandled error in append handler: {e}", exc_info=True)
        # ctx.pubsub.publish_nowait(create_invalid_request_error(f"Error processing chunk {chunk_index}: {e}"))


@event_router.register("input_audio_buffer.commit")
def handle_input_audio_buffer_commit(ctx: SessionContext, event: InputAudioBufferCommitEvent) -> None:
    """Handles commit requests. Creates committed event and new buffer."""
    if event.item_id not in ctx.input_audio_buffers: logger.warning(f"Commit for {event.item_id}, but buffer not found."); return
    input_audio_buffer = ctx.input_audio_buffers[event.item_id]
    logger.info(f"Commit triggered for buffer: {event.item_id} (Duration: {input_audio_buffer.duration_ms} ms)")
    if input_audio_buffer.duration_ms < MIN_AUDIO_BUFFER_DURATION_MS:
        logger.warning(f"Committed buffer {event.item_id} too short. Skipping transcription.")
        ctx.input_audio_buffers.pop(event.item_id, None); new_buf = InputAudioBuffer(ctx.pubsub); ctx.input_audio_buffers[new_buf.id] = new_buf
        logger.info(f"Created new buffer {new_buf.id} after skipping short commit."); return
    else:
        try:
            previous_item_id = None # FIXME: Logic này cần xem lại
            committed_event = InputAudioBufferCommittedEvent(type="input_audio_buffer.committed", item_id=event.item_id, previous_item_id=previous_item_id)
            ctx.pubsub.publish_nowait(committed_event); logger.info(f"Published committed event for {event.item_id}")
            new_buf = InputAudioBuffer(ctx.pubsub); ctx.input_audio_buffers[new_buf.id] = new_buf
            logger.info(f"Created new buffer {new_buf.id} to receive subsequent audio.")
        except Exception as e: logger.error(f"Failed to publish committed event for {event.item_id}: {e}", exc_info=True)

@event_router.register("input_audio_buffer.clear")
def handle_input_audio_buffer_clear(ctx: SessionContext, event: InputAudioBufferClearEvent) -> None:
    """Handles request to clear the current input buffer."""
    logger.info(f"Clear requested: {event}")
    if not ctx.input_audio_buffers: logger.warning("Clear requested but no active buffer."); ctx.pubsub.publish_nowait(InputAudioBufferClearedEvent(type="input_audio_buffer.cleared", item_id=None)); return
    buf_id = next(reversed(ctx.input_audio_buffers)); ctx.input_audio_buffers.pop(buf_id, None); logger.info(f"Cleared buffer: {buf_id}")
    try: cleared_event = InputAudioBufferClearedEvent(type="input_audio_buffer.cleared", item_id=buf_id); ctx.pubsub.publish_nowait(cleared_event)
    except Exception as e: logger.error(f"Failed to publish cleared event: {e}", exc_info=True)
    new_buf = InputAudioBuffer(ctx.pubsub); ctx.input_audio_buffers[new_buf.id] = new_buf; logger.info(f"Created new buffer after clear: {new_buf.id}")

@event_router.register("input_audio_buffer.speech_stopped")
def handle_input_audio_buffer_speech_stopped(ctx: SessionContext, event: InputAudioBufferSpeechStoppedEvent) -> None:
    """Handles VAD stop event, updates buffer state, then triggers commit."""
    logger.info(f"VAD stopped for buffer: {event.item_id} at {event.audio_end_ms} ms. Triggering commit.")
    if event.item_id in ctx.input_audio_buffers:
        input_audio_buffer = ctx.input_audio_buffers[event.item_id]
        if input_audio_buffer.vad_state.audio_end_ms is None:
             input_audio_buffer.vad_state.audio_end_ms = event.audio_end_ms
        else: logger.warning(f"VAD stopped for {event.item_id}, but audio_end_ms was already set ({input_audio_buffer.vad_state.audio_end_ms}). Using existing value.")
    else: logger.warning(f"VAD stopped for {event.item_id}, but buffer not found to update state."); return
    try:
         commit_event = InputAudioBufferCommitEvent(type="input_audio_buffer.commit", item_id=event.item_id)
         ctx.pubsub.publish_nowait(commit_event)
    except Exception as e: logger.error(f"Failed to publish commit event from VAD stop: {e}", exc_info=True)


@event_router.register("input_audio_buffer.committed")
async def handle_input_audio_buffer_committed(ctx: SessionContext, event: InputAudioBufferCommittedEvent) -> None:
    """Handles the committed buffer event: starts transcription and cleans up."""
    if event.item_id not in ctx.input_audio_buffers:
        logger.warning(f"Committed buffer {event.item_id} not found for transcription."); return

    input_audio_buffer = ctx.input_audio_buffers[event.item_id]
    logger.info(f"Processing committed buffer for transcription: {event.item_id} ({input_audio_buffer.duration_ms} ms)")

    audio_data_to_process = input_audio_buffer.data

    # --- BỎ PHẦN LƯU FILE DEBUG ---
    # if audio_data_to_process is not None and audio_data_to_process.size > 0:
    #     try:
    #         # ... code lưu file ...
    #     except Exception as e:
    #         logger.error(f"Failed to save committed audio segment {event.item_id}: {e}", exc_info=True)
    # else: logger.warning(f"No audio data found in committed buffer {event.item_id} to save.")
    # --- KẾT THÚC BỎ PHẦN LƯU FILE ---

    if event.item_id not in ctx.input_audio_buffers: logger.warning(f"Buffer {event.item_id} gone before transcribe start."); return
    input_audio_buffer_for_transcribe = ctx.input_audio_buffers[event.item_id]
    if input_audio_buffer_for_transcribe.data is None or input_audio_buffer_for_transcribe.data.size == 0:
         logger.warning(f"Buffer {event.item_id} is empty when starting transcription. Skipping."); del ctx.input_audio_buffers[event.item_id]; return
    if not ctx.transcription_client: logger.error("No transcription client."); del ctx.input_audio_buffers[event.item_id]; return

    transcriber = InputAudioBufferTranscriber(pubsub=ctx.pubsub, transcription_client=ctx.transcription_client, input_audio_buffer=input_audio_buffer_for_transcribe, session=ctx.session, conversation=ctx.conversation)
    logger.info(f"Starting transcription task for committed buffer: {event.item_id}")
    transcriber.start()
    if transcriber.task is None: logger.error(f"Failed to start transcribe task for {event.item_id}."); del ctx.input_audio_buffers[event.item_id]; return

    try:
        await transcriber.task; logger.info(f"Transcription task completed for buffer: {event.item_id}")
    except Exception as trans_error: logger.error(f"Transcription task failed for {event.item_id}: {trans_error}", exc_info=True)
    finally:
         if event.item_id in ctx.input_audio_buffers: del ctx.input_audio_buffers[event.item_id]; # logger.debug(f"Removed processed buffer {event.item_id}.")
         else: logger.warning(f"Buffer {event.item_id} already removed before finally block.")