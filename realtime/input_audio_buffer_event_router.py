# realtime/input_audio_buffer_event_router.py
import base64
from io import BytesIO
import logging
from typing import Literal
# import soundfile as sf # Bỏ import nếu không lưu file debug
# from pathlib import Path # Bỏ import nếu không lưu file debug
# import os # Bỏ import nếu không lưu file debug
import time
import numpy as np
import librosa # Đảm bảo đã cài: pip install librosa
from faster_whisper.transcribe import get_speech_timestamps
from faster_whisper.vad import VadOptions
from numpy.typing import NDArray
from openai.types.beta.realtime.error_event import Error
from pydantic import ValidationError # Để bắt lỗi validation cụ thể

# Đảm bảo đường dẫn import này đúng với cấu trúc dự án của bạn
# from audio import audio_samples_from_file # Không dùng nữa, thay bằng frombuffer
from realtime.context import SessionContext
from realtime.event_router import EventRouter
from realtime.input_audio_buffer import (
    MAX_VAD_WINDOW_SIZE_SAMPLES,
    MS_SAMPLE_RATE, # Nên là 16 (16000 / 1000)
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

MIN_AUDIO_BUFFER_DURATION_MS = 100  # based on the OpenAI's API response
MAX_BUFFER_DURATION_MS = 5000 # Giới hạn thời gian buffer tối đa (5 giây)

logger = logging.getLogger(__name__)

event_router = EventRouter()
empty_input_audio_buffer_commit_error = Error(
    type="invalid_request_error",
    message="Error committing input audio buffer: the buffer is empty.",
)

type SpeechTimestamp = dict[Literal["start", "end"], int]


# --- Các hàm tiện ích ---
def to_ms_speech_timestamps(speech_timestamps: list[SpeechTimestamp]) -> list[SpeechTimestamp]:
    """Converts speech timestamps from samples (at 16kHz) to milliseconds."""
    # Đầu vào từ get_speech_timestamps đã là samples ở 16kHz
    # MS_SAMPLE_RATE nên là 16 (16000 samples / 1000 ms)
    if not speech_timestamps:
        return []
    converted_timestamps = []
    for ts in speech_timestamps:
        # Kiểm tra kiểu dữ liệu an toàn hơn
        start_samples = ts.get("start")
        end_samples = ts.get("end")
        if isinstance(start_samples, (int, np.integer)) and isinstance(end_samples, (int, np.integer)):
             # Chia cho số sample mỗi ms
             start_ms = start_samples // MS_SAMPLE_RATE
             end_ms = end_samples // MS_SAMPLE_RATE
             converted_timestamps.append({"start": start_ms, "end": end_ms})
        else:
             logger.warning(f"Skipping conversion for invalid timestamp sample format: {ts}")
             # Có thể trả về timestamp gốc hoặc bỏ qua
             # converted_timestamps.append(ts) # Trả về gốc nếu muốn
    return converted_timestamps
# --- Các hàm tiện ích ---
def resample_audio_data(data: NDArray[np.float32], sample_rate: int, target_sample_rate: int) -> NDArray[np.float32]:
    if sample_rate == target_sample_rate: return data
    ratio = target_sample_rate / sample_rate
    target_length = int(len(data) * ratio)
    if target_length == 0 and data.size > 0: target_length = 1
    elif target_length == 0 and data.size == 0: return np.array([], dtype=np.float32)
    return np.interp(np.linspace(0, len(data), target_length, endpoint=False), np.arange(len(data)), data).astype(np.float32)

# --- Logic VAD ---
def vad_detection_flow( input_audio_buffer: InputAudioBuffer, turn_detection: TurnDetection) -> InputAudioBufferSpeechStartedEvent | InputAudioBufferSpeechStoppedEvent | None:
    """Performs VAD on the latest audio window and returns start/stop events."""
    logger.debug(f"VAD Check: threshold={turn_detection.threshold}, silence_duration_ms={turn_detection.silence_duration_ms}")
    if input_audio_buffer.data is None or input_audio_buffer.data.size == 0:
        logger.debug("VAD Skip: Buffer data is empty.")
        return None

    # Tính toán kích thước cửa sổ VAD bằng mili giây (thường là 3000ms)
    vad_window_duration_ms = MAX_VAD_WINDOW_SIZE_SAMPLES // MS_SAMPLE_RATE

    audio_window = input_audio_buffer.data[-MAX_VAD_WINDOW_SIZE_SAMPLES:]

    # Kiểm tra xem cửa sổ có đủ lớn để phát hiện khoảng lặng tối thiểu không
    min_required_samples = turn_detection.silence_duration_ms * MS_SAMPLE_RATE
    if audio_window.size < min_required_samples:
        logger.debug(f"VAD Skip: Window size ({audio_window.size} samples) < min required for silence ({min_required_samples} samples).")
        return None

    try:
        # Lấy timestamp gốc bằng sample rate của buffer (16000)
        raw_speech_timestamps_samples = list(get_speech_timestamps(
            audio_window,
            vad_options=VadOptions(
                threshold=turn_detection.threshold,
                min_silence_duration_ms=turn_detection.silence_duration_ms,
                speech_pad_ms=turn_detection.prefix_padding_ms
            ),
            sampling_rate=INPUT_AUDIO_BUFFER_SAMPLE_RATE # 16000
        ))
        # Chuyển sang ms để dễ debug và so sánh
        speech_timestamps_ms = to_ms_speech_timestamps(raw_speech_timestamps_samples)

    except Exception as e:
        logger.error(f"Error during VAD processing: {e}", exc_info=True)
        return None

    # Log kết quả VAD
    if not speech_timestamps_ms:
         logger.debug("VAD found no speech in window.")
    elif len(speech_timestamps_ms) == 1:
         logger.debug(f"VAD found single speech segment (ms): {speech_timestamps_ms[0]}")
    else: # > 1 segment
         logger.warning(f"VAD found multiple speech segments (ms): {speech_timestamps_ms}")

    last_speech_segment_ms = speech_timestamps_ms[-1] if speech_timestamps_ms else None
    now_ms = input_audio_buffer.duration_ms # Thời lượng buffer hiện tại

    # --- Logic phát hiện Start ---
    if input_audio_buffer.vad_state.audio_start_ms is None:
        if last_speech_segment_ms is not None:
            # Tính thời điểm bắt đầu tương đối trong cửa sổ VAD
            start_in_window_ms = last_speech_segment_ms["start"]
            # Tính thời điểm bắt đầu của cửa sổ VAD so với đầu buffer
            window_start_offset_ms = max(0, now_ms - vad_window_duration_ms)
            # Thời điểm bắt đầu tuyệt đối
            absolute_start_ms = window_start_offset_ms + start_in_window_ms
            # Gán vào state, đảm bảo không âm
            input_audio_buffer.vad_state.audio_start_ms = max(0, absolute_start_ms)
            logger.info(f"VAD detected speech start at ~{input_audio_buffer.vad_state.audio_start_ms} ms (Buffer duration: {now_ms} ms)")
            return InputAudioBufferSpeechStartedEvent(item_id=input_audio_buffer.id, audio_start_ms=input_audio_buffer.vad_state.audio_start_ms)
        else:
            # Chưa có giọng nói trong cửa sổ này, chưa bắt đầu
            return None
    # --- Logic phát hiện Stop ---
    else:
        # Nếu đã bắt đầu nói, kiểm tra điều kiện dừng
        speech_stopped = False

        # Điều kiện 1: Không có giọng nói nào trong cửa sổ hiện tại
        if last_speech_segment_ms is None:
            logger.debug("VAD Stop condition 1: No speech detected in current window.")
            speech_stopped = True

        # Điều kiện 2: Giọng nói cuối cùng kết thúc đủ sớm trước cuối cửa sổ
        # (nghĩa là có khoảng lặng >= silence_duration_ms ở cuối cửa sổ)
        elif last_speech_segment_ms["end"] < (vad_window_duration_ms - turn_detection.silence_duration_ms):
             logger.debug(f"VAD Stop condition 2: Last speech ended at {last_speech_segment_ms['end']}ms, well before window end ({vad_window_duration_ms}ms).")
             speech_stopped = True

        # Nếu một trong các điều kiện dừng được thỏa mãn
        if speech_stopped:
            # Tính toán thời điểm kết thúc cẩn thận
            estimated_end_ms = 0 # Khởi tạo

            # Ưu tiên sử dụng thời điểm kết thúc của đoạn nói cuối cùng nếu có
            if raw_speech_timestamps_samples:
                window_start_in_buffer_samples = max(0, input_audio_buffer.data.size - MAX_VAD_WINDOW_SIZE_SAMPLES)
                last_segment_end_samples = raw_speech_timestamps_samples[-1].get('end', 0)
                absolute_end_samples = window_start_in_buffer_samples + last_segment_end_samples
                estimated_end_ms_from_vad = absolute_end_samples // MS_SAMPLE_RATE
                logger.debug(f"End calc (from VAD): window_start={window_start_in_buffer_samples}, last_end={last_segment_end_samples}, abs_end={absolute_end_samples}, est_ms={estimated_end_ms_from_vad}")
                # Dùng thời điểm VAD tính được, nhưng không sớm hơn start_ms
                estimated_end_ms = max(input_audio_buffer.vad_state.audio_start_ms, estimated_end_ms_from_vad)
            else:
                # Nếu không có giọng nói trong cửa sổ cuối, ước lượng dựa trên thời gian im lặng
                estimated_end_ms_no_vad = max(input_audio_buffer.vad_state.audio_start_ms, now_ms - turn_detection.silence_duration_ms)
                logger.debug(f"End calc (no speech in window): using buffer duration - silence: {estimated_end_ms_no_vad}")
                estimated_end_ms = estimated_end_ms_no_vad

            # Đảm bảo end_ms không sớm hơn start_ms + thời lượng tối thiểu
            estimated_end_ms = max(estimated_end_ms, input_audio_buffer.vad_state.audio_start_ms + MIN_AUDIO_BUFFER_DURATION_MS)
             # Đảm bảo end_ms không vượt quá thời lượng buffer hiện tại (có thể trừ padding)
            estimated_end_ms = min(estimated_end_ms, now_ms - turn_detection.prefix_padding_ms)


            logger.debug(f"Final estimated_end_ms before assignment: {estimated_end_ms}")
            # Chỉ gán nếu giá trị hợp lệ (ví dụ > start_ms)
            if estimated_end_ms > input_audio_buffer.vad_state.audio_start_ms:
                 input_audio_buffer.vad_state.audio_end_ms = estimated_end_ms
                 logger.info(f"VAD detected speech end at ~{input_audio_buffer.vad_state.audio_end_ms} ms (Buffer duration: {now_ms} ms)")
                 return InputAudioBufferSpeechStoppedEvent(item_id=input_audio_buffer.id, audio_end_ms=input_audio_buffer.vad_state.audio_end_ms)
            else:
                 logger.warning(f"Calculated end_ms ({estimated_end_ms}) not valid compared to start_ms ({input_audio_buffer.vad_state.audio_start_ms}). Not stopping yet.")
                 return None
        else:
            # Vẫn đang nói, chưa đủ điều kiện dừng
            return None

# --- Event Handlers ---

def _get_chunk_counter(ctx: SessionContext) -> int:
    """Gets and increments the chunk counter for the session."""
    if not hasattr(ctx, '_chunk_counter'):
         setattr(ctx, '_chunk_counter', 0) # Khởi tạo nếu chưa có
    ctx._chunk_counter += 1
    return ctx._chunk_counter

@event_router.register("input_audio_buffer.append")
def handle_input_audio_buffer_append(ctx: SessionContext, event: InputAudioBufferAppendEvent) -> None:
    """Handles incoming audio chunks, reads, resamples, appends, checks max duration, runs VAD."""
    chunk_index = _get_chunk_counter(ctx)
    session_id = ctx.session.id
    INCOMING_SR = 24000 # Sample rate client gửi đến
    TARGET_SR = INPUT_AUDIO_BUFFER_SAMPLE_RATE # 16000 - Sample rate buffer nội bộ
    logger.debug(f"Handling append event for session {session_id}, chunk {chunk_index}")

    try:
        # --- Bước 1: Decode Base64 ---
        audio_bytes = base64.b64decode(event.audio)
        if not audio_bytes:
            logger.warning(f"Chunk {chunk_index}: Empty audio after base64.")
            return

        # --- Bước 2: Đọc Bytes thành Numpy Array (Giả định 24kHz, s16le) ---
        try:
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            # Chuyển sang float32 trong [-1, 1]
            audio_chunk_raw_24k = audio_int16.astype(np.float32) / 32768.0
            if audio_chunk_raw_24k.size == 0:
                logger.warning(f"Chunk {chunk_index}: Empty audio after frombuffer.")
                return
            # logger.debug(f"Chunk {chunk_index}: Decoded raw 24k data - Shape: {audio_chunk_raw_24k.shape}, dtype: {audio_chunk_raw_24k.dtype}")
        except Exception as read_err:
            logger.error(f"Chunk {chunk_index}: Failed to read bytes into numpy array: {read_err}", exc_info=True)
            return

        # --- Bước 4: Resample xuống 16kHz (Dùng librosa) ---
        try:
             if INCOMING_SR != TARGET_SR:
                 audio_chunk_16k = librosa.resample(y=audio_chunk_raw_24k, orig_sr=INCOMING_SR, target_sr=TARGET_SR, res_type='soxr_hq')
                 # logger.debug(f"Chunk {chunk_index}: Resampled 16k data (librosa) - Shape: {audio_chunk_16k.shape}, dtype: {audio_chunk_16k.dtype}")
             else:
                  audio_chunk_16k = audio_chunk_raw_24k
                  # logger.debug(f"Chunk {chunk_index}: Skipping resampling.")

             if audio_chunk_16k.size == 0:
                 logger.warning(f"Chunk {chunk_index}: Empty audio after resampling.")
                 return
        except Exception as resample_err:
             logger.error(f"Chunk {chunk_index}: Failed to resample using librosa: {resample_err}", exc_info=True)
             return # Bỏ qua nếu resample lỗi

        # --- Bước 6: Append vào Buffer chính ---
        if not ctx.input_audio_buffers:
            logger.warning("No active input audio buffer found. Creating a new one.")
            new_buffer = InputAudioBuffer(ctx.pubsub)
            ctx.input_audio_buffers[new_buffer.id] = new_buffer
            input_audio_buffer_id = new_buffer.id
        else:
            input_audio_buffer_id = next(reversed(ctx.input_audio_buffers))

        # Kiểm tra xem buffer có tồn tại không trước khi truy cập
        if input_audio_buffer_id not in ctx.input_audio_buffers:
            logger.error(f"Buffer {input_audio_buffer_id} not found in context, cannot append chunk {chunk_index}.")
            # Có thể tạo lại buffer ở đây nếu cần thiết
            new_buffer = InputAudioBuffer(ctx.pubsub); ctx.input_audio_buffers[new_buffer.id] = new_buffer; input_audio_buffer_id = new_buffer.id
            input_audio_buffer = new_buffer
            logger.warning(f"Recreated buffer {input_audio_buffer_id} due to missing previous buffer.")
        else:
             input_audio_buffer = ctx.input_audio_buffers[input_audio_buffer_id]

        input_audio_buffer.append(audio_chunk_16k)
        logger.debug(f"Chunk {chunk_index}: Appended 16k data to buffer {input_audio_buffer_id}. New duration: {input_audio_buffer.duration_ms} ms")

        # --- KIỂM TRA THỜI LƯỢNG TỐI ĐA TRƯỚC KHI CHẠY VAD ---
        # Chỉ kiểm tra nếu VAD chưa dừng cho buffer này
        if input_audio_buffer.vad_state.audio_end_ms is None and \
           input_audio_buffer.duration_ms > MAX_BUFFER_DURATION_MS:

            logger.warning(f"Buffer {input_audio_buffer_id} duration ({input_audio_buffer.duration_ms}ms) exceeded max duration {MAX_BUFFER_DURATION_MS}ms. Forcing commit.")
            # Gán tạm thời điểm kết thúc để tránh VAD chạy lại sau khi commit
            # input_audio_buffer.vad_state.audio_end_ms = input_audio_buffer.duration_ms # Đánh dấu là đã xử lý
            # Publish sự kiện commit - **SỬA LỖI PYDANTIC**
            try:
                commit_event = InputAudioBufferCommitEvent(
                    type="input_audio_buffer.commit", # <<< Thêm trường type
                    item_id=input_audio_buffer_id
                )
                ctx.pubsub.publish_nowait(commit_event)
            except ValidationError as e:
                 logger.error(f"Failed to create InputAudioBufferCommitEvent: {e}")
            except Exception as pub_err:
                 logger.error(f"Failed to publish forced commit event: {pub_err}")
            # Không cần chạy VAD nữa cho buffer này, handler commit sẽ tạo buffer mới
            return

        # --- Bước 7: Chạy VAD (chỉ khi chưa vượt quá max duration và VAD chưa stop) ---
        if input_audio_buffer.vad_state.audio_end_ms is None and \
           ctx.session.turn_detection is not None and \
           ctx.session.turn_detection.type == "server_vad":
             vad_event = vad_detection_flow(input_audio_buffer, ctx.session.turn_detection)
             if vad_event is not None:
                 logger.debug(f"Chunk {chunk_index}: VAD produced event: {vad_event.type}")
                 ctx.pubsub.publish_nowait(vad_event)

    except Exception as e:
        logger.error(f"Chunk {chunk_index}: Unhandled error in input_audio_buffer.append: {e}", exc_info=True)
        ctx.pubsub.publish_nowait(create_invalid_request_error(f"Error processing chunk {chunk_index}: {e}"))


@event_router.register("input_audio_buffer.commit")
def handle_input_audio_buffer_commit(ctx: SessionContext, event: InputAudioBufferCommitEvent) -> None:
    """Handles commit requests (from VAD stop or forced duration). Creates committed event and new buffer."""
    if event.item_id not in ctx.input_audio_buffers:
         logger.warning(f"Commit requested for buffer {event.item_id}, but it's not found (possibly already processed or cleared).")
         return

    input_audio_buffer = ctx.input_audio_buffers[event.item_id]
    logger.info(f"Commit triggered for buffer: {event.item_id} (Duration: {input_audio_buffer.duration_ms} ms)")

    # Kiểm tra thời lượng tối thiểu trước khi publish `committed`
    if input_audio_buffer.duration_ms < MIN_AUDIO_BUFFER_DURATION_MS:
        logger.warning(f"Committed buffer {event.item_id} too short ({input_audio_buffer.duration_ms}ms). Skipping transcription.")
        # Vẫn xóa buffer cũ và tạo buffer mới để hệ thống tiếp tục
        ctx.input_audio_buffers.pop(event.item_id, None) # Xóa an toàn
        new_input_audio_buffer = InputAudioBuffer(ctx.pubsub)
        ctx.input_audio_buffers[new_input_audio_buffer.id] = new_input_audio_buffer
        logger.info(f"Created new buffer {new_input_audio_buffer.id} after skipping short commit for {event.item_id}.")
        # Gửi lỗi cho client nếu cần thiết
        # ctx.pubsub.publish_nowait(create_invalid_request_error(f"...buffer too small..."))
        return
    else:
        # Publish sự kiện `committed` để bắt đầu quá trình phiên mã
        try:
            # Xác định previous_item_id một cách an toàn hơn
            # Note: Logic này có thể cần xem lại tùy thuộc vào cách items được quản lý
            previous_item_id = None
            if ctx.conversation and ctx.conversation.items:
                 # Lấy id của item cuối cùng đã hoàn thành (nếu có)
                 # Cần cẩn thận hơn nếu item cuối có thể chưa commit xong
                 # Ví dụ đơn giản: lấy item cuối cùng
                 # previous_item_id = next(reversed(ctx.conversation.items), None)
                 pass # Tạm thời bỏ qua để tránh lỗi logic phức tạp

            committed_event = InputAudioBufferCommittedEvent(
                type="input_audio_buffer.committed", # <<< Thêm trường type
                item_id=event.item_id,
                previous_item_id=previous_item_id # FIXME: Logic lấy previous_item_id
            )
            ctx.pubsub.publish_nowait(committed_event)
            logger.info(f"Published committed event for buffer {event.item_id}")

            # Tạo buffer mới NGAY LẬP TỨC để các chunk tiếp theo đi vào đó
            # Buffer cũ (event.item_id) sẽ được xử lý bởi handle_input_audio_buffer_committed
            new_input_audio_buffer = InputAudioBuffer(ctx.pubsub)
            ctx.input_audio_buffers[new_input_audio_buffer.id] = new_input_audio_buffer
            logger.info(f"Created new buffer {new_input_audio_buffer.id} to receive subsequent audio.")

        except ValidationError as e:
            logger.error(f"Failed to create InputAudioBufferCommittedEvent for {event.item_id}: {e}")
        except Exception as pub_err:
            logger.error(f"Failed to publish committed event for {event.item_id}: {pub_err}")

@event_router.register("input_audio_buffer.clear")
def handle_input_audio_buffer_clear(ctx: SessionContext, event: InputAudioBufferClearEvent) -> None:
    """Handles request to clear the current input buffer."""
    logger.info(f"Clear requested. Event details: {event}") # Log event để xem có item_id không
    # Logic này giả định clear luôn xóa buffer cuối cùng
    if not ctx.input_audio_buffers:
        logger.warning("Clear requested but no active buffer found.")
        # Vẫn nên publish cleared event theo spec?
        ctx.pubsub.publish_nowait(InputAudioBufferClearedEvent(type="input_audio_buffer.cleared", item_id=None)) # Gửi với item_id=None
        return

    input_audio_buffer_id = next(reversed(ctx.input_audio_buffers))
    if input_audio_buffer_id in ctx.input_audio_buffers:
        ctx.input_audio_buffers.pop(input_audio_buffer_id)
        logger.info(f"Cleared buffer: {input_audio_buffer_id}")
        try:
            # Publish sự kiện cleared
            cleared_event = InputAudioBufferClearedEvent(
                type="input_audio_buffer.cleared", # <<< Thêm type
                item_id=input_audio_buffer_id
            )
            ctx.pubsub.publish_nowait(cleared_event)
        except ValidationError as e:
             logger.error(f"Failed to create InputAudioBufferClearedEvent: {e}")
        except Exception as pub_err:
             logger.error(f"Failed to publish cleared event: {pub_err}")
    else:
         logger.warning(f"Attempted to clear buffer {input_audio_buffer_id}, but it was already removed.")
         # Vẫn có thể publish event cleared với id nếu muốn
         # ctx.pubsub.publish_nowait(...)

    # Luôn tạo buffer mới sau khi clear
    input_audio_buffer = InputAudioBuffer(ctx.pubsub)
    ctx.input_audio_buffers[input_audio_buffer.id] = input_audio_buffer
    logger.info(f"Created new buffer after clear: {input_audio_buffer.id}")

@event_router.register("input_audio_buffer.speech_stopped")
def handle_input_audio_buffer_speech_stopped(ctx: SessionContext, event: InputAudioBufferSpeechStoppedEvent) -> None:
    """Handles VAD stop event by triggering a commit."""
    logger.info(f"VAD stopped for buffer: {event.item_id}. Triggering commit.")
    # Publish commit event, không tạo buffer mới ở đây nữa
    try:
         commit_event = InputAudioBufferCommitEvent(
              type="input_audio_buffer.commit", # <<< Thêm type
              item_id=event.item_id
         )
         ctx.pubsub.publish_nowait(commit_event)
    except ValidationError as e:
        logger.error(f"Failed to create InputAudioBufferCommitEvent from VAD stop: {e}")
    except Exception as pub_err:
        logger.error(f"Failed to publish commit event from VAD stop: {pub_err}")
    # Buffer mới sẽ được tạo bởi handle_input_audio_buffer_commit

@event_router.register("input_audio_buffer.committed")
async def handle_input_audio_buffer_committed(ctx: SessionContext, event: InputAudioBufferCommittedEvent) -> None:
    """Handles the committed buffer event: starts transcription and cleans up."""
    if event.item_id not in ctx.input_audio_buffers:
        # Buffer có thể đã bị xóa bởi clear hoặc xử lý xong rất nhanh
        logger.warning(f"Committed buffer {event.item_id} not found when trying to transcribe (might be already processed or cleared).")
        return

    input_audio_buffer = ctx.input_audio_buffers[event.item_id]
    logger.info(f"Processing committed buffer for transcription: {event.item_id} ({input_audio_buffer.duration_ms} ms)")

    # Kiểm tra lại phòng trường hợp buffer rỗng dù đã qua check MIN_AUDIO_BUFFER_DURATION_MS
    if input_audio_buffer.data is None or input_audio_buffer.data.size == 0:
        logger.warning(f"Buffer {event.item_id} is empty when starting transcription. Skipping.")
        if event.item_id in ctx.input_audio_buffers:
             del ctx.input_audio_buffers[event.item_id] # Dọn dẹp buffer rỗng
        return

    # Kiểm tra transcription client
    if not ctx.transcription_client:
        logger.error("No transcription client configured. Cannot transcribe.")
        if event.item_id in ctx.input_audio_buffers:
             del ctx.input_audio_buffers[event.item_id] # Dọn dẹp
        return

    # Tạo và chạy transcriber
    # Lưu ý: InputAudioBufferTranscriber cần nhận đúng đối tượng buffer
    transcriber = InputAudioBufferTranscriber(
        pubsub=ctx.pubsub,
        transcription_client=ctx.transcription_client,
        input_audio_buffer=input_audio_buffer, # Truyền buffer object
        session=ctx.session,
        conversation=ctx.conversation,
    )
    logger.info(f"Starting transcription task for committed buffer: {event.item_id}")
    transcriber.start()

    if transcriber.task is None:
        logger.error(f"Failed to start transcribe task for {event.item_id}.")
        if event.item_id in ctx.input_audio_buffers:
             del ctx.input_audio_buffers[event.item_id] # Dọn dẹp
        return

    # Chờ task hoàn thành và dọn dẹp buffer
    try:
        await transcriber.task
        logger.info(f"Transcription task completed for buffer: {event.item_id}")
    except Exception as trans_error:
        logger.error(f"Transcription task failed for buffer {event.item_id}: {trans_error}", exc_info=True)
    finally:
         # Luôn xóa buffer khỏi context sau khi xử lý xong (hoặc lỗi)
         if event.item_id in ctx.input_audio_buffers:
             del ctx.input_audio_buffers[event.item_id]
             logger.debug(f"Removed processed buffer {event.item_id} from context.")
         else:
              logger.warning(f"Buffer {event.item_id} was already removed before finally block in committed handler.")