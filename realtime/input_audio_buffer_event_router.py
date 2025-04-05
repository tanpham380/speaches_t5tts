import base64
from io import BytesIO
import logging
from typing import Literal

from faster_whisper.transcribe import get_speech_timestamps
from faster_whisper.vad import VadOptions
import numpy as np
from numpy.typing import NDArray
from openai.types.beta.realtime.error_event import Error

from  audio import audio_samples_from_file
from  realtime.context import SessionContext
from  realtime.event_router import EventRouter
from  realtime.input_audio_buffer import (
    MAX_VAD_WINDOW_SIZE_SAMPLES,
    MIN_VAD_WINDOW_SIZE_SAMPLES,
    MS_SAMPLE_RATE,
    InputAudioBuffer,
    InputAudioBufferTranscriber,
)
from  mouble_types.realtime import (
    InputAudioBufferAppendEvent,
    InputAudioBufferClearedEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferCommittedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    TurnDetection,
    create_invalid_request_error,
    create_server_error,
)

MIN_AUDIO_BUFFER_DURATION_MS = 100  # based on the OpenAI's API response

logger = logging.getLogger(__name__)

event_router = EventRouter()

empty_input_audio_buffer_commit_error = Error(
    type="invalid_request_error",
    message="Error committing input audio buffer: the buffer is empty.",
)

type SpeechTimestamp = dict[Literal["start", "end"], int]


# NOTE: `signal.resample_poly` **might** be a better option for resampling audio data
# TODO: also found in src/speaches/audio.py. Remove duplication
def resample_audio_data(data: NDArray[np.float32], sample_rate: int, target_sample_rate: int) -> NDArray[np.float32]:
    ratio = target_sample_rate / sample_rate
    target_length = int(len(data) * ratio)
    return np.interp(np.linspace(0, len(data), target_length), np.arange(len(data)), data).astype(np.float32)


# TODO: also found in src/speaches/routers/vad.py. Remove duplication
def to_ms_speech_timestamps(speech_timestamps: list[SpeechTimestamp]) -> list[SpeechTimestamp]:
    for i in range(len(speech_timestamps)):
        speech_timestamps[i]["start"] = speech_timestamps[i]["start"] // MS_SAMPLE_RATE
        speech_timestamps[i]["end"] = speech_timestamps[i]["end"] // MS_SAMPLE_RATE
    return speech_timestamps


# Trong realtime/input_audio_buffer_event_router.py

def vad_detection_flow(
    input_audio_buffer: InputAudioBuffer, turn_detection: TurnDetection
) -> InputAudioBufferSpeechStartedEvent | InputAudioBufferSpeechStoppedEvent | None:
    # Lấy cửa sổ audio đủ lớn để VAD hoạt động tốt
    # MAX_VAD_WINDOW_SIZE_SAMPLES nên đủ lớn, ví dụ 1-2 giây audio (16000-32000 samples)
    audio_window = input_audio_buffer.data[-MAX_VAD_WINDOW_SIZE_SAMPLES:]
    if len(audio_window) < MIN_VAD_WINDOW_SIZE_SAMPLES: # Thêm một hằng số MIN nếu cần
         logger.debug("Audio window too short for VAD, skipping.")
         return None

    # logger.debug(f"Running VAD on window size: {len(audio_window)}")
    logger.debug(f"[{input_audio_buffer.id}] Running VAD on window size: {len(audio_window)} samples ({len(audio_window)/16.0:.1f} ms)")

    try:
        vad_options = VadOptions(
            threshold=turn_detection.threshold,
            min_silence_duration_ms=turn_detection.silence_duration_ms,
            speech_pad_ms=turn_detection.prefix_padding_ms,
        )
        logger.debug(f"[{input_audio_buffer.id}] VAD Options: threshold={vad_options.threshold}, min_silence={vad_options.min_silence_duration_ms}, padding={vad_options.speech_pad_ms}")
        raw_timestamps = get_speech_timestamps(audio_window, vad_options=vad_options)
        logger.debug(f"[{input_audio_buffer.id}] Raw VAD timestamps (samples): {raw_timestamps}")
        speech_timestamps = to_ms_speech_timestamps(raw_timestamps)


        speech_timestamps = to_ms_speech_timestamps(
            get_speech_timestamps(
                audio_window,
                vad_options=vad_options,
                # sample_rate=16000 # Đảm bảo sample_rate đúng nếu cần
            )

 
        )
        logger.debug(f"[{input_audio_buffer.id}] VAD detected speech_timestamps (ms): {speech_timestamps}")


    except Exception as e:
        logger.exception(f"Error calling get_speech_timestamps: {e}")
        return None # Trả về None nếu VAD lỗi

    # logger.debug(f"Raw speech timestamps (samples): {speech_timestamps}") # Log timestamp gốc (samples) nếu cần debug VAD

    # Lấy kết quả cuối cùng (nếu có)
    speech_timestamp = speech_timestamps[-1] if speech_timestamps else None
    # logger.debug(f"Last speech timestamp (ms): {speech_timestamp}")

    # --- Logic phát hiện trạng thái ---
    is_currently_speaking = speech_timestamp is not None
    was_previously_speaking = input_audio_buffer.vad_state.audio_start_ms is not None

    # logger.debug(f"VAD State: is_currently_speaking={is_currently_speaking}, was_previously_speaking={was_previously_speaking}")

    if not was_previously_speaking and is_currently_speaking:
        # --- Bắt đầu nói ---
        # Tính toán thời gian bắt đầu dựa trên timestamp trả về và vị trí cửa sổ
        # Cần cẩn thận vì timestamp trả về là tương đối với audio_window
        relative_start_ms = speech_timestamp["start"] # Đây là ms trong audio_window
        window_start_offset_ms = input_audio_buffer.duration_ms - (len(audio_window) // MS_SAMPLE_RATE)
        absolute_start_ms = window_start_offset_ms + relative_start_ms

        input_audio_buffer.vad_state.audio_start_ms = absolute_start_ms
        logger.info(f"[{input_audio_buffer.id}] Speech STARTED detected at ~{absolute_start_ms}ms")
        return InputAudioBufferSpeechStartedEvent(
            item_id=input_audio_buffer.id,
            audio_start_ms=absolute_start_ms,
        )
    elif was_previously_speaking and not is_currently_speaking:
        # --- Dừng nói ---
        # Tính thời gian kết thúc. Có thể lấy thời điểm hiện tại của buffer trừ đi padding
        # Hoặc dựa vào timestamp cuối cùng nếu có cách xác định chính xác hơn
        absolute_end_ms = input_audio_buffer.duration_ms - turn_detection.prefix_padding_ms # Giữ nguyên cách tính này có vẻ hợp lý
        # Đảm bảo end > start
        absolute_end_ms = max(absolute_end_ms, input_audio_buffer.vad_state.audio_start_ms)

        # Lưu lại trạng thái kết thúc và reset trạng thái bắt đầu cho lần nói tiếp theo
        input_audio_buffer.vad_state.audio_end_ms = absolute_end_ms
        previous_start_time = input_audio_buffer.vad_state.audio_start_ms
        input_audio_buffer.vad_state.audio_start_ms = None # Reset trạng thái

        logger.info(f"[{input_audio_buffer.id}] Speech STOPPED detected at ~{absolute_end_ms}ms (started at {previous_start_time}ms)")
        return InputAudioBufferSpeechStoppedEvent(
            item_id=input_audio_buffer.id,
            audio_end_ms=absolute_end_ms,
        )
    # elif was_previously_speaking and is_currently_speaking:
        # Vẫn đang nói, không cần làm gì
        # logger.debug("Still speaking...")
        # pass
    # elif not was_previously_speaking and not is_currently_speaking:
         # Vẫn đang im lặng
         # logger.debug("Still silent...")
         # pass

    return None # Không có thay đổi trạng thái


# Client Events


@event_router.register("input_audio_buffer.append")
def handle_input_audio_buffer_append(ctx: SessionContext, event: InputAudioBufferAppendEvent) -> None:
    
    try:
        audio_chunk = audio_samples_from_file(BytesIO(base64.b64decode(event.audio)))
        # convert the audio data from 24kHz (sample rate defined in the API spec) to 16kHz (sample rate used by the VAD and for transcription)
        audio_chunk = resample_audio_data(audio_chunk, 24000, 16000)
        input_audio_buffer_id = next(reversed(ctx.input_audio_buffers))
        logger.debug(f"[{ctx.session.id}] Calling transcription client...")

        input_audio_buffer = ctx.input_audio_buffers[input_audio_buffer_id]
        input_audio_buffer.append(audio_chunk)

        # --- XÓA HOẶC COMMENT LẠI PHẦN DEBUG NÀY ---
        # logger.warning(f"[{ctx.session.id}] DEBUG: Force committing buffer {input_audio_buffer_id}")
        # ctx.pubsub.publish_nowait(
        #      InputAudioBufferCommittedEvent(
        #          previous_item_id=None,
        #          item_id=input_audio_buffer_id,
        #      )
        # )
        # --- KẾT THÚC PHẦN DEBUG ---

        # Bật lại VAD flow
        if ctx.session.turn_detection is not None:
            vad_event = vad_detection_flow(input_audio_buffer, ctx.session.turn_detection)
            if vad_event is not None:
                ctx.pubsub.publish_nowait(vad_event)
    except Exception as e:
        logger.exception(f"[{ctx.session.id}] Error in handle_input_audio_buffer_append: {e}")
        error_event = create_server_error(f"Error processing audio chunk: {e}")
        ctx.pubsub.publish_nowait(error_event)


@event_router.register("input_audio_buffer.commit")
def handle_input_audio_buffer_commit(ctx: SessionContext, event: InputAudioBufferCommitEvent) -> None:
    # Lấy ID buffer từ sự kiện client gửi
    input_audio_buffer_id = event.item_id # Giả sử client gửi đúng ID

    logger.info(f"[{ctx.session.id}] Received client commit for item {input_audio_buffer_id}.")

    # Kiểm tra xem buffer có tồn tại không
    if input_audio_buffer_id not in ctx.input_audio_buffers:
        logger.warning(f"[{ctx.session.id}] Client committed non-existent buffer {input_audio_buffer_id}. Ignoring.")
        ctx.pubsub.publish_nowait(create_invalid_request_error(f"Cannot commit buffer: ID {input_audio_buffer_id} not found."))
        return

    input_audio_buffer = ctx.input_audio_buffers[input_audio_buffer_id]

    if input_audio_buffer.duration_ms < MIN_AUDIO_BUFFER_DURATION_MS:
        logger.warning(f"[{ctx.session.id}] Client committed buffer {input_audio_buffer_id} which is too short ({input_audio_buffer.duration_ms}ms). Ignoring.")
        ctx.pubsub.publish_nowait(
            create_invalid_request_error(f"Buffer {input_audio_buffer_id} too small to commit.")
        )
        # Không tạo buffer mới nếu commit không thành công
    else:
        logger.info(f"[{ctx.session.id}] Committing buffer {input_audio_buffer_id} due to client request.")
        # Publish sự kiện commit
        ctx.pubsub.publish_nowait(
            InputAudioBufferCommittedEvent(
                previous_item_id=next(reversed(ctx.conversation.items), None),  # FIXME
                item_id=input_audio_buffer_id,
            )
        )
        # Tạo buffer mới cho audio tiếp theo
        new_buffer = InputAudioBuffer(ctx.pubsub)
        ctx.input_audio_buffers[new_buffer.id] = new_buffer
        logger.info(f"[{ctx.session.id}] Created new buffer {new_buffer.id} after client commit.")
        # Reset trạng thái VAD của buffer cũ (nếu cần thiết và buffer cũ không bị xóa ngay)
        input_audio_buffer.is_speaking = False
        input_audio_buffer.last_speech_start_time = None

@event_router.register("input_audio_buffer.clear")
def handle_input_audio_buffer_clear(ctx: SessionContext, _event: InputAudioBufferClearEvent) -> None:
    ctx.input_audio_buffers.popitem()
    # OpenAI's doesn't send an error if the buffer is already empty.
    ctx.pubsub.publish_nowait(InputAudioBufferClearedEvent())
    input_audio_buffer = InputAudioBuffer(ctx.pubsub)
    ctx.input_audio_buffers[input_audio_buffer.id] = input_audio_buffer


# Server Events


@event_router.register("input_audio_buffer.speech_stopped")
def handle_input_audio_buffer_speech_stopped(ctx: SessionContext, event: InputAudioBufferSpeechStoppedEvent) -> None:
    # Tạo một buffer mới sẵn sàng cho audio tiếp theo sau khi VAD dừng
    new_input_audio_buffer = InputAudioBuffer(ctx.pubsub) # Đổi tên biến để rõ ràng hơn
    ctx.input_audio_buffers[new_input_audio_buffer.id] = new_input_audio_buffer
    # Publish sự kiện commit cho buffer cũ đã kết thúc bởi VAD
    ctx.pubsub.publish_nowait(
        InputAudioBufferCommittedEvent(
            previous_item_id=next(reversed(ctx.conversation.items), None),  # FIXME: Cần logic lấy previous_item_id đúng
            item_id=event.item_id, # item_id của buffer vừa kết thúc
        )
    )
    logger.info(f"[{ctx.session.id}] Speech stopped, committed buffer {event.item_id} and created new buffer {new_input_audio_buffer.id}")

# --- CHỈ GIỮ LẠI MỘT PHIÊN BẢN CỦA HÀM NÀY ---
@event_router.register("input_audio_buffer.committed")
async def handle_input_audio_buffer_committed(ctx: SessionContext, event: InputAudioBufferCommittedEvent) -> None:
    logger.info(f"[{ctx.session.id}] Handler handle_input_audio_buffer_committed called for item {event.item_id}.")
    # Kiểm tra xem item_id có tồn tại trong buffers không
    if event.item_id not in ctx.input_audio_buffers:
        logger.error(f"[{ctx.session.id}] Committed item_id {event.item_id} not found in input_audio_buffers.")
        # Cân nhắc publish lỗi về client
        ctx.pubsub.publish_nowait(create_server_error(f"Internal error: Committed audio buffer {event.item_id} not found."))
        return

    input_audio_buffer = ctx.input_audio_buffers[event.item_id]
    logger.debug(f"[{ctx.session.id}] Found input audio buffer {event.item_id} with duration {input_audio_buffer.duration_ms}ms.")

    # Kiểm tra lại xem buffer có thực sự có dữ liệu không (phòng trường hợp)
    if input_audio_buffer.duration_ms < MIN_AUDIO_BUFFER_DURATION_MS: # Hoặc kiểm tra len(input_audio_buffer.data) > 0
         logger.warning(f"[{ctx.session.id}] Committed buffer {event.item_id} has insufficient duration ({input_audio_buffer.duration_ms}ms). Skipping transcription.")
         # Có thể cần xóa buffer này khỏi dict nếu không dùng nữa
         # del ctx.input_audio_buffers[event.item_id]
         # Cân nhắc publish sự kiện hoàn thành với transcript rỗng
         # completion_event = ConversationItemInputAudioTranscriptionCompletedEvent(item_id=event.item_id, transcript="")
         # ctx.pubsub.publish_nowait(completion_event)
         return

    logger.debug(f"[{ctx.session.id}] Initializing InputAudioBufferTranscriber for item {event.item_id}.")
    transcriber = InputAudioBufferTranscriber(
        pubsub=ctx.pubsub,
        transcription_client=ctx.transcription_client,
        input_audio_buffer=input_audio_buffer,
        session=ctx.session,
        conversation=ctx.conversation, # Truyền conversation state nếu transcriber cần
    )

    logger.debug(f"[{ctx.session.id}] Starting transcriber task for item {event.item_id}.")
    transcriber.start() # Bắt đầu task chạy nền

    # Không nên await trực tiếp ở đây nếu bạn muốn event loop tiếp tục xử lý các sự kiện khác
    # Việc await transcriber.task sẽ block handler này cho đến khi phiên âm xong.
    # Thay vào đó, transcriber.start() nên tạo task chạy nền.
    # await transcriber.task # <<< BỎ DÒNG NÀY NẾU START() TẠO TASK NỀN

    # Nếu bạn cần chờ kết quả ở đây (không khuyến khích vì block), bạn cần await task
    if transcriber.task:
         logger.debug(f"[{ctx.session.id}] Waiting for transcriber task {event.item_id} to complete (this might block)...")
         try:
            await transcriber.task # Chỉ await nếu bạn thực sự cần kết quả ngay lập tức trong handler này
            logger.info(f"[{ctx.session.id}] Transcriber task {event.item_id} completed in handler.")
         except Exception as e:
             logger.exception(f"[{ctx.session.id}] Error awaiting transcriber task {event.item_id}: {e}")
    else:
        logger.warning(f"[{ctx.session.id}] Transcriber task for {event.item_id} was not created or already done.")

    # Logic dọn dẹp buffer cũ có thể thực hiện ở đây hoặc trong transcriber khi hoàn thành
    # Ví dụ: del ctx.input_audio_buffers[event.item_id]
