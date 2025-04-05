# Giả sử trong file realtime/input_audio_buffer.py hoặc file tương tự

import asyncio
import logging
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
# <<< THÊM IMPORT SOUNDFILE >>>
try:
    import soundfile as sf
except ImportError:
    logging.getLogger(__name__).error("Soundfile not installed. Please install using: pip install soundfile")
    # Hoặc định nghĩa sf.write giả để tránh lỗi runtime ngay lập tức
    class DummySF:
        def write(*args, **kwargs): pass
    sf = DummySF()

from openai import NotGiven
from pydantic import BaseModel

# Import các thành phần cần thiết
from realtime.utils import generate_item_id, task_done_callback, generate_event_id
from mouble_types.realtime import (
    ConversationItemContentInputAudio,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemMessage,
    ConversationItemCreatedEvent, # Thêm nếu muốn gửi
    ServerEvent,
    Session,
    ErrorEvent,
    create_server_error,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    # <<< KIỂU CLIENT OPENAI >>>
    from openai.resources.audio import AsyncTranscriptions
    from realtime.conversation_event_router import Conversation
    from realtime.pubsub import EventPubSub

# --- Các định nghĩa khác giữ nguyên ---
SAMPLE_RATE = 16000
MS_SAMPLE_RATE = 16
MAX_VAD_WINDOW_SIZE_SAMPLES = 3000 * MS_SAMPLE_RATE
MIN_VAD_WINDOW_SIZE_SAMPLES = 16000 // 2

class VadState(BaseModel):
    audio_start_ms: int | None = None
    audio_end_ms: int | None = None

class InputAudioBuffer:
    def __init__(self, pubsub: EventPubSub) -> None:
        self.id = generate_item_id()
        self.data: NDArray[np.float32] = np.array([], dtype=np.float32) # Giữ float32 để ghi WAV
        self.vad_state = VadState()
        self.pubsub = pubsub
        self.is_speaking: bool = False
        self.last_speech_start_time: float | None = None

    @property
    def size(self) -> int: return len(self.data)
    @property
    def duration(self) -> float: return len(self.data) / SAMPLE_RATE
    @property
    def duration_ms(self) -> int: return len(self.data) // MS_SAMPLE_RATE

    def append(self, audio_chunk: NDArray[np.float32]) -> None:
        self.data = np.append(self.data, audio_chunk)

    def get_full_audio_data_16k(self) -> NDArray[np.float32] | None:
        """Trả về dữ liệu audio đã áp dụng VAD (nếu có) hoặc toàn bộ."""
        # Lấy dữ liệu dựa trên VAD state hoặc toàn bộ nếu VAD chưa chạy/xác định
        if self.vad_state.audio_start_ms is None or self.vad_state.audio_end_ms is None:
            audio_to_process = self.data
        else:
            start_sample = max(0, self.vad_state.audio_start_ms * MS_SAMPLE_RATE)
            end_sample = min(len(self.data), self.vad_state.audio_end_ms * MS_SAMPLE_RATE)
            if start_sample >= end_sample:
                 audio_to_process = np.array([], dtype=np.float32)
            else:
                 audio_to_process = self.data[start_sample:end_sample]

        if len(audio_to_process) > 0:
             return audio_to_process.copy()
        return None

    @property
    def data_w_vad_applied(self) -> NDArray[np.float32]:
        # Giữ lại property này vì transcriber đang dùng
        data = self.get_full_audio_data_16k()
        return data if data is not None else np.array([], dtype=np.float32)

# --- LỚP TRANSCRIBER (PHIÊN BẢN DÙNG OPENAI CLIENT, KHÔNG CÓ DELTA) ---
logger = logging.getLogger(__name__)

class InputAudioBufferTranscriber:
    def __init__(
        self,
        *,
        pubsub: EventPubSub,
        transcription_client: AsyncTranscriptions, # <<< Kiểu OpenAI Client
        input_audio_buffer: InputAudioBuffer,
        session: Session,
        conversation: Conversation,
    ) -> None:
        self.pubsub = pubsub
        self.transcription_client = transcription_client
        self.input_audio_buffer = input_audio_buffer
        self.session = session
        self.conversation = conversation
        self.task: asyncio.Task[None] | None = None
        self.item_id = input_audio_buffer.id
        logger.info(f"[{self.session.id}] InputAudioBufferTranscriber initialized for item {self.item_id}.")

    async def _handler(self) -> None:
        """Task chạy nền để phiên âm toàn bộ buffer và publish kết quả cuối cùng."""
        logger.info(f"[{self.session.id}] Transcription task for item {self.item_id} starting execution.")
        item_id_for_event = self.item_id
        transcript = "" # Khởi tạo transcript rỗng

        try:
            # 1. Tạo/Cập nhật Conversation Item
            content_item = ConversationItemContentInputAudio(transcript=None, type="input_audio")
            item = ConversationItemMessage(
                id=item_id_for_event,
                role="user",
                content=[content_item],
                status="incomplete",
            )
            previous_item_id = self.conversation.get_last_item_id()
            try:
                self.conversation.create_item(item)
                created_event = ConversationItemCreatedEvent(item=item, previous_item_id=previous_item_id)
                logger.debug(f"[{self.session.id}] Publishing {created_event.type} for item {item.id}")
                self.pubsub.publish_nowait(created_event)
            except ValueError:
                 logger.warning(f"[{self.session.id}] Item {item.id} already exists, updating status.")
                 existing_item = self.conversation.get_item(item.id)
                 if existing_item and isinstance(existing_item, ConversationItemMessage):
                      existing_item.status = "incomplete"
                      if existing_item.content and isinstance(existing_item.content[0], ConversationItemContentInputAudio):
                           existing_item.content[0].transcript = None

            # 2. Lấy dữ liệu audio và chuẩn bị file
            audio_data_to_transcribe = self.input_audio_buffer.data_w_vad_applied # Dùng property này
            if audio_data_to_transcribe is None or len(audio_data_to_transcribe) < (SAMPLE_RATE * 0.1):
                 logger.warning(f"[{self.session.id}] Audio data for item {item_id_for_event} is empty or too short after VAD. Finishing.")
                 transcript = "" # Giữ transcript rỗng
            else:
                 logger.debug(f"[{self.session.id}] Preparing {len(audio_data_to_transcribe)} audio samples (16kHz) for transcription API.")
                 # Ghi vào BytesIO để gửi cho API OpenAI
                 file_like_object = BytesIO()
                 try:
                     # Dùng float32 khi ghi WAV cho OpenAI API (theo khuyến nghị của họ)
                     sf.write(
                         file_like_object,
                         audio_data_to_transcribe,
                         samplerate=SAMPLE_RATE, # 16000
                         subtype="FLOAT", # <<< Gửi float thay vì PCM_16
                         format="WAV",
                     )
                     file_like_object.seek(0) # Đưa con trỏ về đầu file
                     logger.debug(f"[{self.session.id}] Prepared WAV data in BytesIO.")
                 except Exception as write_error:
                      logger.exception(f"[{self.session.id}] Error writing audio data to BytesIO: {write_error}")
                      raise RuntimeError("Failed to prepare audio data for API") from write_error


                 # 3. Gọi API OpenAI create
                 logger.info(f"[{self.session.id}] Calling transcription_client.create for item {item_id_for_event}...")
                 start_time = time.time()
                 try:
                     # Tuple (filename, fileobject) là cần thiết cho thư viện openai khi gửi file
                     file_tuple = ("audio.wav", file_like_object)
                     transcription_response = await self.transcription_client.create(
                         file=file_tuple,
                         model=self.session.input_audio_transcription.model, # Model trong session config
                         response_format="text", # Yêu cầu text đơn giản
                         language=self.session.input_audio_transcription.language or NotGiven(),
                         # temperature=self.session.temperature # Có thể thêm nếu API hỗ trợ
                     )
                     # API trả về trực tiếp chuỗi transcript khi response_format="text"
                     transcript = transcription_response if isinstance(transcription_response, str) else ""

                 except Exception as api_error:
                      logger.exception(f"[{self.session.id}] Error calling OpenAI transcription API: {api_error}")
                      raise RuntimeError(f"OpenAI API call failed: {api_error}") from api_error

                 end_time = time.time()
                 logger.info(f"[{self.session.id}] OpenAI transcription API call finished in {end_time - start_time:.3f} seconds.")


            # 4. Publish sự kiện hoàn thành (luôn luôn, kể cả khi transcript rỗng)
            final_transcript = transcript.strip() if transcript else ""
            completion_event = ConversationItemInputAudioTranscriptionCompletedEvent(
                item_id=item_id_for_event,
                transcript=final_transcript
            )
            logger.info(f"[{self.session.id}] Publishing {completion_event.type} for item {item_id_for_event}. Final transcript: '{final_transcript}'")
            self.pubsub.publish_nowait(completion_event)

            # Cập nhật trạng thái và nội dung cuối cùng cho item
            item.status = "completed"
            content_item.transcript = final_transcript

        except Exception as e:
            logger.exception(f"[{self.session.id}] Error during transcription task for item {self.item_id}: {e}")
            error_event = create_server_error(f"Transcription failed for item {self.item_id}: {e}")
            logger.debug(f"[{self.session.id}] Publishing error event: {error_event.error.message}")
            self.pubsub.publish_nowait(error_event)
            if 'item' in locals() and hasattr(item, 'status'):
                 try:
                      item_to_update = self.conversation.get_item(self.item_id)
                      if item_to_update: item_to_update.status = "failed"
                 except Exception: pass
        finally:
             logger.info(f"[{self.session.id}] Transcription task for item {self.item_id} finished execution.")

    def start(self) -> None:
        """Bắt đầu task phiên âm chạy nền."""
        if self.task is None or self.task.done():
            logger.info(f"[{self.session.id}] Creating transcription task for item {self.item_id}.")
            self.task = asyncio.create_task(self._handler())
            self.task.add_done_callback(task_done_callback)
        else:
            logger.warning(f"[{self.session.id}] Transcription task for item {self.item_id} is already running or pending.")