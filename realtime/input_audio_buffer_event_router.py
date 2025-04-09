# #realtime/input_audio_buffer_event_router.py
# import base64
# from io import BytesIO
# import logging
# from typing import Literal
# import soundfile as sf

# from faster_whisper.transcribe import get_speech_timestamps
# from faster_whisper.vad import VadOptions
# import numpy as np
# from numpy.typing import NDArray
# from openai.types.beta.realtime.error_event import Error

# from audio import audio_samples_from_file # Assuming this path is correct
# from  realtime.context import SessionContext
# from  realtime.event_router import EventRouter
# from  realtime.input_audio_buffer import (
#     MAX_VAD_WINDOW_SIZE_SAMPLES,
#     MS_SAMPLE_RATE,
#     InputAudioBuffer,
#     InputAudioBufferTranscriber,
# )
# from  mouble_types.realtime import (
#     InputAudioBufferAppendEvent,
#     InputAudioBufferClearedEvent,
#     InputAudioBufferClearEvent,
#     InputAudioBufferCommitEvent,
#     InputAudioBufferCommittedEvent,
#     InputAudioBufferSpeechStartedEvent,
#     InputAudioBufferSpeechStoppedEvent,
#     TurnDetection,
#     create_invalid_request_error,
# )

# MIN_AUDIO_BUFFER_DURATION_MS = 100  # based on the OpenAI's API response

# logger = logging.getLogger(__name__)

# event_router = EventRouter()
# empty_input_audio_buffer_commit_error = Error(
#     type="invalid_request_error",
#     message="Error committing input audio buffer: the buffer is empty.",
# )

# type SpeechTimestamp = dict[Literal["start", "end"], int]


# # NOTE: `signal.resample_poly` **might** be a better option for resampling audio data
# # TODO: also found in src/speaches/audio.py. Remove duplication
# def resample_audio_data(data: NDArray[np.float32], sample_rate: int, target_sample_rate: int) -> NDArray[np.float32]:
#     ratio = target_sample_rate / sample_rate
#     target_length = int(len(data) * ratio)
#     return np.interp(np.linspace(0, len(data), target_length), np.arange(len(data)), data).astype(np.float32)


# # TODO: also found in src/speaches/routers/vad.py. Remove duplication
# def to_ms_speech_timestamps(speech_timestamps: list[SpeechTimestamp]) -> list[SpeechTimestamp]:
#     for i in range(len(speech_timestamps)):
#         speech_timestamps[i]["start"] = speech_timestamps[i]["start"] // MS_SAMPLE_RATE
#         speech_timestamps[i]["end"] = speech_timestamps[i]["end"] // MS_SAMPLE_RATE
#     return speech_timestamps


# def vad_detection_flow(
#     input_audio_buffer: InputAudioBuffer, turn_detection: TurnDetection
# ) -> InputAudioBufferSpeechStartedEvent | InputAudioBufferSpeechStoppedEvent | None:
#     audio_window = input_audio_buffer.data[-MAX_VAD_WINDOW_SIZE_SAMPLES:]

#     speech_timestamps = to_ms_speech_timestamps(
#         get_speech_timestamps(
#             audio_window,
#             vad_options=VadOptions(
#                 threshold=turn_detection.threshold,
#                 min_silence_duration_ms=turn_detection.silence_duration_ms,
#                 speech_pad_ms=turn_detection.prefix_padding_ms,
#             ),
#         )
#     )
#     if len(speech_timestamps) > 1:
#         logger.warning(f"More than one speech timestamp: {speech_timestamps}")

#     speech_timestamp = speech_timestamps[-1] if len(speech_timestamps) > 0 else None

#     # logger.debug(f"Speech timestamps: {speech_timestamps}")
#     if input_audio_buffer.vad_state.audio_start_ms is None:
#         if speech_timestamp is None:
#             return None
#         input_audio_buffer.vad_state.audio_start_ms = (
#             input_audio_buffer.duration_ms - len(audio_window) // MS_SAMPLE_RATE + speech_timestamp["start"]
#         )
#         return InputAudioBufferSpeechStartedEvent(
#             item_id=input_audio_buffer.id,
#             audio_start_ms=input_audio_buffer.vad_state.audio_start_ms,
#         )

#     else:  # noqa: PLR5501
#         if speech_timestamp is None:
#             # TODO: not quite correct. dependent on window size
#             input_audio_buffer.vad_state.audio_end_ms = (
#                 input_audio_buffer.duration_ms - turn_detection.prefix_padding_ms
#             )
#             return InputAudioBufferSpeechStoppedEvent(
#                 item_id=input_audio_buffer.id,
#                 audio_end_ms=input_audio_buffer.vad_state.audio_end_ms,
#             )

#         elif speech_timestamp["end"] < 3000 and input_audio_buffer.duration_ms > 3000:  # FIX: magic number
#             input_audio_buffer.vad_state.audio_end_ms = (
#                 input_audio_buffer.duration_ms - turn_detection.prefix_padding_ms
#             )

#             return InputAudioBufferSpeechStoppedEvent(
#                 item_id=input_audio_buffer.id,
#                 audio_end_ms=input_audio_buffer.vad_state.audio_end_ms,
#             )

#     return None


# # Client Events


# @event_router.register("input_audio_buffer.append")
# def handle_input_audio_buffer_append(ctx: SessionContext, event: InputAudioBufferAppendEvent) -> None:
#     audio_chunk = audio_samples_from_file(BytesIO(base64.b64decode(event.audio)))
#     # convert the audio data from 24kHz (sample rate defined in the API spec) to 16kHz (sample rate used by the VAD and for transcription)
#     audio_chunk = resample_audio_data(audio_chunk, 24000, 16000)
#     input_audio_buffer_id = next(reversed(ctx.input_audio_buffers))
#     input_audio_buffer = ctx.input_audio_buffers[input_audio_buffer_id]
#     input_audio_buffer.append(audio_chunk)
#     if ctx.session.turn_detection is not None:
#         vad_event = vad_detection_flow(input_audio_buffer, ctx.session.turn_detection)
#         if vad_event is not None:
#             ctx.pubsub.publish_nowait(vad_event)


# @event_router.register("input_audio_buffer.commit")
# def handle_input_audio_buffer_commit(ctx: SessionContext, _event: InputAudioBufferCommitEvent) -> None:
#     input_audio_buffer_id = next(reversed(ctx.input_audio_buffers))
#     input_audio_buffer = ctx.input_audio_buffers[input_audio_buffer_id]
#     if input_audio_buffer.duration_ms < MIN_AUDIO_BUFFER_DURATION_MS:
#         ctx.pubsub.publish_nowait(
#             create_invalid_request_error(
#                 message=f"Error committing input audio buffer: buffer too small. Expected at least {MIN_AUDIO_BUFFER_DURATION_MS}ms of audio, but buffer only has {input_audio_buffer.duration_ms}.00ms of audio."
#             )
#         )
#     else:
#         ctx.pubsub.publish_nowait(
#             InputAudioBufferCommittedEvent(
#                 previous_item_id=next(reversed(ctx.conversation.items), None),  # FIXME
#                 item_id=input_audio_buffer_id,
#             )
#         )
#         input_audio_buffer = InputAudioBuffer(ctx.pubsub)
#         ctx.input_audio_buffers[input_audio_buffer.id] = input_audio_buffer


# @event_router.register("input_audio_buffer.clear")
# def handle_input_audio_buffer_clear(ctx: SessionContext, _event: InputAudioBufferClearEvent) -> None:
#     ctx.input_audio_buffers.popitem()
#     # OpenAI's doesn't send an error if the buffer is already empty.
#     ctx.pubsub.publish_nowait(InputAudioBufferClearedEvent())
#     input_audio_buffer = InputAudioBuffer(ctx.pubsub)
#     ctx.input_audio_buffers[input_audio_buffer.id] = input_audio_buffer


# # Server Events


# @event_router.register("input_audio_buffer.speech_stopped")
# def handle_input_audio_buffer_speech_stopped(ctx: SessionContext, event: InputAudioBufferSpeechStoppedEvent) -> None:
#     input_audio_buffer = InputAudioBuffer(ctx.pubsub)
#     ctx.input_audio_buffers[input_audio_buffer.id] = input_audio_buffer
#     ctx.pubsub.publish_nowait(
#         InputAudioBufferCommittedEvent(
#             previous_item_id=next(reversed(ctx.conversation.items), None),  # FIXME
#             item_id=event.item_id,
#         )
#     )


# @event_router.register("input_audio_buffer.committed")
# async def handle_input_audio_buffer_committed(ctx: SessionContext, event: InputAudioBufferCommittedEvent) -> None:
#     input_audio_buffer = ctx.input_audio_buffers[event.item_id]

#     transcriber = InputAudioBufferTranscriber(
#         pubsub=ctx.pubsub,
#         transcription_client=ctx.transcription_client,
#         input_audio_buffer=input_audio_buffer,
#         session=ctx.session,
#         conversation=ctx.conversation,
#     )
#     transcriber.start()
#     assert transcriber.task is not None
#     await transcriber.task

# realtime/input_audio_buffer_event_router.py
import base64
from io import BytesIO
import logging
from typing import Literal
import soundfile as sf # Để ghi file WAV
from pathlib import Path # Để xử lý đường dẫn file/thư mục
import os # Để tạo thư mục
import time # <<< Thêm time để tạo timestamp cho tên file

from faster_whisper.transcribe import get_speech_timestamps
from faster_whisper.vad import VadOptions
import numpy as np
from numpy.typing import NDArray
from openai.types.beta.realtime.error_event import Error

# Đảm bảo đường dẫn import này đúng với cấu trúc dự án của bạn
from audio import audio_samples_from_file
from realtime.context import SessionContext
from realtime.event_router import EventRouter
from realtime.input_audio_buffer import (
    MAX_VAD_WINDOW_SIZE_SAMPLES,
    MS_SAMPLE_RATE, # Nên là 16 nếu sample rate buffer là 16k
    InputAudioBuffer,
    InputAudioBufferTranscriber,
    SAMPLE_RATE as INPUT_AUDIO_BUFFER_SAMPLE_RATE, # Lấy sample rate (16000) từ module buffer
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

MIN_AUDIO_BUFFER_DURATION_MS = 100  # based on the OpenAI's API response

logger = logging.getLogger(__name__)

event_router = EventRouter()
empty_input_audio_buffer_commit_error = Error(
    type="invalid_request_error",
    message="Error committing input audio buffer: the buffer is empty.",
)

type SpeechTimestamp = dict[Literal["start", "end"], int]

# --- Định nghĩa các thư mục lưu audio DEBUG ---
DEBUG_AUDIO_SAVE_DIR = Path("./debug_audio")
DEBUG_CHUNK_RAW_DIR = DEBUG_AUDIO_SAVE_DIR / "1_received_24k"
DEBUG_CHUNK_RESAMPLED_DIR = DEBUG_AUDIO_SAVE_DIR / "2_resampled_16k"
VAD_AUDIO_SAVE_DIR = DEBUG_AUDIO_SAVE_DIR / "3_vad_committed_16k" # <<< Thư mục lưu VAD cuối

# Tạo các thư mục nếu chưa tồn tại
try:
    DEBUG_CHUNK_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_CHUNK_RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    VAD_AUDIO_SAVE_DIR.mkdir(parents=True, exist_ok=True) # <<< Đảm bảo thư mục VAD cũng trong debug
    logger.info(f"Debug audio will be saved under: {DEBUG_AUDIO_SAVE_DIR.resolve()}")
except OSError as e:
     logger.error(f"Could not create directories under {DEBUG_AUDIO_SAVE_DIR}: {e}")

# --- Các hàm tiện ích ---
def resample_audio_data(data: NDArray[np.float32], sample_rate: int, target_sample_rate: int) -> NDArray[np.float32]:
    # ...(Giữ nguyên hàm resample hiện tại hoặc thay bằng librosa nếu muốn)...
    if sample_rate == target_sample_rate: return data
    ratio = target_sample_rate / sample_rate
    target_length = int(len(data) * ratio)
    if target_length == 0 and data.size > 0: target_length = 1
    elif target_length == 0 and data.size == 0: return np.array([], dtype=np.float32)
    return np.interp(np.linspace(0, len(data), target_length, endpoint=False), np.arange(len(data)), data).astype(np.float32)

def to_ms_speech_timestamps(speech_timestamps: list[SpeechTimestamp]) -> list[SpeechTimestamp]:
    # ...(Giữ nguyên)...
    if not speech_timestamps or isinstance(speech_timestamps[0].get("start"), float) or speech_timestamps[0].get("start", 0) < 1000: return speech_timestamps
    converted_timestamps = []
    for ts in speech_timestamps:
        start_samples, end_samples = ts.get("start"), ts.get("end")
        if isinstance(start_samples, int) and isinstance(end_samples, int):
             converted_timestamps.append({"start": start_samples // MS_SAMPLE_RATE, "end": end_samples // MS_SAMPLE_RATE})
        else:
             logger.warning(f"Skipping conversion for invalid timestamp format: {ts}"); converted_timestamps.append(ts)
    return converted_timestamps

# --- Logic VAD ---
def vad_detection_flow( input_audio_buffer: InputAudioBuffer, turn_detection: TurnDetection) -> InputAudioBufferSpeechStartedEvent | InputAudioBufferSpeechStoppedEvent | None:
    # ...(Giữ nguyên)...
    if input_audio_buffer.data is None or input_audio_buffer.data.size == 0: return None
    audio_window = input_audio_buffer.data[-MAX_VAD_WINDOW_SIZE_SAMPLES:]
    if audio_window.size == 0: return None
    try:
        raw_speech_timestamps = get_speech_timestamps(audio_window, vad_options=VadOptions(threshold=turn_detection.threshold, min_silence_duration_ms=turn_detection.silence_duration_ms, speech_pad_ms=turn_detection.prefix_padding_ms), sampling_rate=INPUT_AUDIO_BUFFER_SAMPLE_RATE)
        speech_timestamps = to_ms_speech_timestamps(list(raw_speech_timestamps))
    except Exception as e: logger.error(f"Error during VAD processing: {e}", exc_info=True); return None
    if len(speech_timestamps) > 1: logger.warning(f"VAD found multiple speech segments: {speech_timestamps}")
    speech_timestamp = speech_timestamps[-1] if speech_timestamps else None
    if input_audio_buffer.vad_state.audio_start_ms is None:
        if speech_timestamp is not None:
            window_duration_ms = len(audio_window) // MS_SAMPLE_RATE
            absolute_start_ms = (input_audio_buffer.duration_ms - window_duration_ms) + speech_timestamp["start"]
            input_audio_buffer.vad_state.audio_start_ms = max(0, absolute_start_ms)
            logger.info(f"VAD detected speech start at ~{input_audio_buffer.vad_state.audio_start_ms} ms")
            return InputAudioBufferSpeechStartedEvent(item_id=input_audio_buffer.id, audio_start_ms=input_audio_buffer.vad_state.audio_start_ms)
        else: return None
    else:
        if speech_timestamp is None:
            input_audio_buffer.vad_state.audio_end_ms = max(input_audio_buffer.vad_state.audio_start_ms, input_audio_buffer.duration_ms - turn_detection.prefix_padding_ms)
            logger.info(f"VAD detected speech end at ~{input_audio_buffer.vad_state.audio_end_ms} ms")
            return InputAudioBufferSpeechStoppedEvent(item_id=input_audio_buffer.id, audio_end_ms=input_audio_buffer.vad_state.audio_end_ms)
        else:
             MAX_UTTERANCE_GUESS_MS = 30000; is_stuck_early = speech_timestamp["end"] < (MAX_VAD_WINDOW_SIZE_SAMPLES // MS_SAMPLE_RATE - turn_detection.silence_duration_ms); is_buffer_long = input_audio_buffer.duration_ms > MAX_UTTERANCE_GUESS_MS
             if is_stuck_early and is_buffer_long:
                 input_audio_buffer.vad_state.audio_end_ms = max(input_audio_buffer.vad_state.audio_start_ms, input_audio_buffer.duration_ms - turn_detection.prefix_padding_ms)
                 logger.warning(f"Forcing speech stop due to potentially stuck VAD at ~{input_audio_buffer.vad_state.audio_end_ms} ms")
                 return InputAudioBufferSpeechStoppedEvent(item_id=input_audio_buffer.id, audio_end_ms=input_audio_buffer.vad_state.audio_end_ms)
             else: return None
    return None

# --- Event Handlers ---

# Biến đếm chunk cho mỗi session (lưu trong context)
def _get_chunk_counter(ctx: SessionContext) -> int:
    if not hasattr(ctx, '_chunk_counter'):
         ctx._chunk_counter = 0
    ctx._chunk_counter += 1
    return ctx._chunk_counter

@event_router.register("input_audio_buffer.append")
def handle_input_audio_buffer_append(ctx: SessionContext, event: InputAudioBufferAppendEvent) -> None:
    """Handles incoming audio chunks, saves intermediate steps, resamples, appends, and runs VAD."""
    chunk_index = _get_chunk_counter(ctx) # Lấy số thứ tự chunk
    session_id = ctx.session.id
    logger.debug(f"Handling append event for session {session_id}, chunk {chunk_index}")

    try:
        # --- Bước 1: Decode Base64 ---
        audio_bytes = base64.b64decode(event.audio)
        if not audio_bytes:
             logger.warning(f"Chunk {chunk_index}: Received empty audio data after base64 decode.")
             return

        # --- Bước 2: Chuyển Bytes thành Numpy Array (24kHz) ---
        # Giả định audio_samples_from_file trả về float32
        audio_chunk_raw_24k = audio_samples_from_file(BytesIO(audio_bytes))
        if audio_chunk_raw_24k.size == 0:
             logger.warning(f"Chunk {chunk_index}: Audio data is empty after decoding from bytes.")
             return
        logger.debug(f"Chunk {chunk_index}: Decoded raw 24k data - Shape: {audio_chunk_raw_24k.shape}, dtype: {audio_chunk_raw_24k.dtype}")


        # --- Bước 3: LƯU FILE TRUNG GIAN (24kHz) ---
        try:
            filename_24k = f"{session_id}_chunk_{chunk_index:04d}_received_24k.wav"
            save_path_24k = DEBUG_CHUNK_RAW_DIR / filename_24k
            # Lưu dưới dạng PCM 16 bit để dễ nghe
            sf.write(str(save_path_24k), audio_chunk_raw_24k, samplerate=24000, format='WAV', subtype='PCM_16')
            logger.debug(f"Chunk {chunk_index}: Saved raw 24k audio to {save_path_24k}")
        except Exception as save_err:
             logger.error(f"Chunk {chunk_index}: Failed to save raw 24k audio: {save_err}")
        # --- Kết thúc Lưu file 24k ---


        # --- Bước 4: Resample xuống 16kHz ---
        audio_chunk_16k = resample_audio_data(audio_chunk_raw_24k, 24000, INPUT_AUDIO_BUFFER_SAMPLE_RATE)
        if audio_chunk_16k.size == 0:
             logger.warning(f"Chunk {chunk_index}: Audio data is empty after resampling to 16k.")
             return
        logger.debug(f"Chunk {chunk_index}: Resampled 16k data - Shape: {audio_chunk_16k.shape}, dtype: {audio_chunk_16k.dtype}")


        # --- Bước 5: LƯU FILE TRUNG GIAN (16kHz) ---
        try:
            filename_16k = f"{session_id}_chunk_{chunk_index:04d}_resampled_16k.wav"
            save_path_16k = DEBUG_CHUNK_RESAMPLED_DIR / filename_16k
            # Lưu dưới dạng PCM 16 bit
            sf.write(str(save_path_16k), audio_chunk_16k, samplerate=INPUT_AUDIO_BUFFER_SAMPLE_RATE, format='WAV', subtype='PCM_16')
            logger.debug(f"Chunk {chunk_index}: Saved resampled 16k audio to {save_path_16k}")
        except Exception as save_err:
             logger.error(f"Chunk {chunk_index}: Failed to save resampled 16k audio: {save_err}")
        # --- Kết thúc Lưu file 16k ---


        # --- Bước 6: Append vào Buffer chính ---
        if not ctx.input_audio_buffers:
             logger.warning("No active input audio buffer found. Creating a new one.")
             new_buffer = InputAudioBuffer(ctx.pubsub); ctx.input_audio_buffers[new_buffer.id] = new_buffer; input_audio_buffer_id = new_buffer.id
        else:
             input_audio_buffer_id = next(reversed(ctx.input_audio_buffers))
        input_audio_buffer = ctx.input_audio_buffers[input_audio_buffer_id]
        input_audio_buffer.append(audio_chunk_16k)
        logger.debug(f"Chunk {chunk_index}: Appended 16k data to buffer {input_audio_buffer_id}. New duration: {input_audio_buffer.duration_ms} ms")


        # --- Bước 7: Chạy VAD ---
        if ctx.session.turn_detection is not None and ctx.session.turn_detection.type == "server_vad":
            vad_event = vad_detection_flow(input_audio_buffer, ctx.session.turn_detection)
            if vad_event is not None:
                logger.debug(f"Chunk {chunk_index}: VAD produced event: {vad_event.type}")
                ctx.pubsub.publish_nowait(vad_event)

    except Exception as e:
        logger.error(f"Chunk {chunk_index}: Error handling input_audio_buffer.append: {e}", exc_info=True)
        ctx.pubsub.publish_nowait(create_invalid_request_error(f"Failed to process appended audio chunk {chunk_index}: {e}"))


# --- Các handler khác giữ nguyên logic lưu file VAD cuối cùng ---

@event_router.register("input_audio_buffer.commit")
def handle_input_audio_buffer_commit(ctx: SessionContext, event: InputAudioBufferCommitEvent) -> None:
    if not ctx.input_audio_buffers: logger.error("Commit req but no buffer"); ctx.pubsub.publish_nowait(empty_input_audio_buffer_commit_error); return
    input_audio_buffer_id = next(reversed(ctx.input_audio_buffers))
    input_audio_buffer = ctx.input_audio_buffers[input_audio_buffer_id]
    logger.info(f"Client explicitly committed buffer: {input_audio_buffer_id} ({input_audio_buffer.duration_ms} ms)")
    if input_audio_buffer.duration_ms < MIN_AUDIO_BUFFER_DURATION_MS:
        logger.warning(f"Committed buffer {input_audio_buffer_id} too short"); ctx.pubsub.publish_nowait(create_invalid_request_error(f"...buffer too small... has {input_audio_buffer.duration_ms}.00ms..."))
    else:
        ctx.pubsub.publish_nowait(InputAudioBufferCommittedEvent(previous_item_id=next(reversed(ctx.conversation.items), None), item_id=input_audio_buffer_id))
        new_input_audio_buffer = InputAudioBuffer(ctx.pubsub); ctx.input_audio_buffers[new_input_audio_buffer.id] = new_input_audio_buffer
        logger.info(f"Created new buffer {new_input_audio_buffer.id} after client commit.")

@event_router.register("input_audio_buffer.clear")
def handle_input_audio_buffer_clear(ctx: SessionContext, _event: InputAudioBufferClearEvent) -> None:
    # ...(Giữ nguyên)...
    if not ctx.input_audio_buffers: logger.warning("Clear req but no buffer")
    else: input_audio_buffer_id = next(reversed(ctx.input_audio_buffers)); ctx.input_audio_buffers.pop(input_audio_buffer_id); logger.info(f"Cleared buffer: {input_audio_buffer_id}"); ctx.pubsub.publish_nowait(InputAudioBufferClearedEvent(item_id=input_audio_buffer_id))
    input_audio_buffer = InputAudioBuffer(ctx.pubsub); ctx.input_audio_buffers[input_audio_buffer.id] = input_audio_buffer
    logger.info(f"Created new buffer after clear: {input_audio_buffer.id}")

@event_router.register("input_audio_buffer.speech_stopped")
def handle_input_audio_buffer_speech_stopped(ctx: SessionContext, event: InputAudioBufferSpeechStoppedEvent) -> None:
    # ...(Giữ nguyên)...
    logger.info(f"VAD stopped for buffer: {event.item_id}. Triggering commit.")
    ctx.pubsub.publish_nowait(InputAudioBufferCommittedEvent(previous_item_id=next(reversed(ctx.conversation.items), None), item_id=event.item_id))
    new_input_audio_buffer = InputAudioBuffer(ctx.pubsub); ctx.input_audio_buffers[new_input_audio_buffer.id] = new_input_audio_buffer
    logger.info(f"Created new buffer {new_input_audio_buffer.id} after VAD stop.")

@event_router.register("input_audio_buffer.committed")
async def handle_input_audio_buffer_committed(ctx: SessionContext, event: InputAudioBufferCommittedEvent) -> None:
    """Handles the committed buffer: saves final audio, starts transcription."""
    if event.item_id not in ctx.input_audio_buffers:
        logger.error(f"Committed buffer {event.item_id} not found. Cannot process.")
        return

    input_audio_buffer = ctx.input_audio_buffers[event.item_id]
    logger.info(f"Processing committed buffer: {event.item_id} ({input_audio_buffer.duration_ms} ms)")

    # --- LƯU FILE VAD CUỐI CÙNG (16kHz) ---
    audio_data_to_save = input_audio_buffer.data
    if audio_data_to_save is not None and audio_data_to_save.size > 0:
        try:
            VAD_AUDIO_SAVE_DIR.mkdir(parents=True, exist_ok=True) # Đảm bảo thư mục tồn tại
            session_id = ctx.session.id
            filename = f"{session_id}_{event.item_id}_vad_committed_16k.wav" # <<< Đổi tên file rõ ràng hơn
            save_path = VAD_AUDIO_SAVE_DIR / filename
            sample_rate_to_save = INPUT_AUDIO_BUFFER_SAMPLE_RATE # 16000

            logger.info(f"Saving final VAD segment {event.item_id} [{input_audio_buffer.duration_ms} ms] to {save_path}")
            sf.write(str(save_path), audio_data_to_save, samplerate=sample_rate_to_save, format='WAV', subtype='PCM_16')
            logger.info(f"Successfully saved final VAD segment to: {save_path}")

        except Exception as e:
            logger.error(f"Failed to save final VAD audio segment {event.item_id}: {e}", exc_info=True)
    else:
        logger.warning(f"No audio data found in committed buffer {event.item_id} to save.")
    if event.item_id not in ctx.input_audio_buffers: logger.error(f"Buffer {event.item_id} gone before transcribe."); return
    input_audio_buffer_for_transcribe = ctx.input_audio_buffers[event.item_id]
    if not ctx.transcription_client: logger.error("No transcription client."); del ctx.input_audio_buffers[event.item_id]; return
    transcriber = InputAudioBufferTranscriber(pubsub=ctx.pubsub, transcription_client=ctx.transcription_client, input_audio_buffer=input_audio_buffer_for_transcribe, session=ctx.session, conversation=ctx.conversation)
    logger.info(f"Starting transcription for committed buffer: {event.item_id}")
    transcriber.start()
    if transcriber.task is None: logger.error(f"Failed to start transcribe task for {event.item_id}."); del ctx.input_audio_buffers[event.item_id]; return
    try: await transcriber.task; logger.info(f"Transcription task completed for buffer: {event.item_id}")
    except Exception as trans_error: logger.error(f"Transcription task failed for buffer {event.item_id}: {trans_error}", exc_info=True)
    finally:
         if event.item_id in ctx.input_audio_buffers: del ctx.input_audio_buffers[event.item_id]; logger.debug(f"Removed buffer {event.item_id}.")