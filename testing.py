import asyncio
import json
import logging
import time
import ssl
import os
import random # Để tạo khoảng dừng ngẫu nhiên

import aiohttp
import numpy as np
# <<< THÊM IMPORT SOUNDFILE >>>
try:
    import soundfile as sf
except ImportError:
    print("Lỗi: Thư viện soundfile chưa được cài đặt.")
    print("Vui lòng cài đặt bằng lệnh: pip install soundfile")
    exit(1)
# <<< KẾT THÚC IMPORT SOUNDFILE >>>

from aiortc import (
    MediaStreamError,
    MediaStreamTrack,
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    RTCDataChannel
)
from av import AudioFrame

try:
    # Import các loại sự kiện cần thiết từ file định nghĩa type của bạn
    from mouble_types.realtime import (
        SessionUpdateEvent, ResponseCreateEvent, PartialSession,
        client_event_type_adapter, server_event_type_adapter, NotGiven, NOT_GIVEN,
        SessionCreatedEvent, ResponseTextDeltaEvent,
        ResponseAudioTranscriptDeltaEvent,
        ConversationItemInputAudioTranscriptionCompletedEvent, ErrorEvent,
        # InputAudioBufferCommitEvent # Không cần commit từ client nữa
    )
except ImportError as e:
    print(f"Lỗi import mouble_types.realtime: {e}")
    print("Vui lòng đảm bảo file mouble_types/realtime.py có thể truy cập được và đúng cấu trúc.")
    exit(1)

# --- Cấu hình ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("realtime_live_test_client")

SERVER_URL = "http://localhost:8000" # Thay đổi nếu cần
API_ENDPOINT = f"{SERVER_URL}/v1/realtime"
MODEL_ID = "Systran/faster-whisper-large-v3" # Model STT bạn muốn test
AUDIO_FILE_PATH = "generated_429000_long.wav" # File audio nguồn

# --- Cấu hình giả lập nói ---
CHUNK_DURATION_S = 1.5  # Độ dài mỗi chunk audio gửi đi (giây)
MIN_PAUSE_S = 0.5     # Khoảng dừng tối thiểu giữa các chunk (giây)
MAX_PAUSE_S = 2.0     # Khoảng dừng tối đa giữa các chunk (giây)
TARGET_SAMPLE_RATE = 48000 # Sample rate gửi qua WebRTC (thường là 48k cho Opus)
TARGET_CHANNELS = 1        # Gửi kênh mono

# Kiểm tra file tồn tại
if not os.path.exists(AUDIO_FILE_PATH):
    logger.error(f"File âm thanh không tìm thấy tại: {AUDIO_FILE_PATH}")
    exit(1)

# Cấu hình WebRTC
ICE_SERVERS = [] # Thêm STUN/TURN server nếu cần test qua mạng phức tạp
RTC_CONFIG = RTCConfiguration(iceServers=[RTCIceServer(**s) for s in ICE_SERVERS])

# Biến toàn cục để lưu session ID (đơn giản hóa cho ví dụ)
current_session_id = None

# --- Hàm trợ giúp ---
async def wait_for_connection(pc: RTCPeerConnection):
    """Chờ cho đến khi kết nối ICE thành công."""
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE Connection State is {pc.iceConnectionState}")
        # Không raise lỗi trực tiếp ở đây để tránh unhandled exception
        # Hàm chờ bên dưới sẽ xử lý timeout hoặc trạng thái failed/closed

    # Chờ trạng thái ổn định hoặc lỗi
    t_start = time.time()
    timeout = 20.0 # Đặt timeout cho ICE connection (giây)
    while time.time() - t_start < timeout:
        state = pc.iceConnectionState
        if state in ["completed", "connected"]:
            logger.info("ICE Connection Established.")
            return # Thoát khi thành công
        if state in ["failed", "closed", "disconnected"]:
             logger.error(f"ICE Connection failed or closed prematurely ({state})")
             raise ConnectionError(f"ICE Connection failed or closed prematurely ({state})")
        await asyncio.sleep(0.1) # Chờ và kiểm tra lại

    # Nếu hết timeout
    logger.error(f"ICE Connection timed out after {timeout} seconds (state: {pc.iceConnectionState}).")
    raise ConnectionError("ICE Connection timed out")

# --- Track Âm thanh tùy chỉnh để gửi Chunk ---
class AudioChunkTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, file_path, chunk_duration_s, target_sample_rate, target_channels):
        super().__init__()
        self.file_path = file_path
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.chunk_samples = int(chunk_duration_s * target_sample_rate)
        self.dtype = np.int16 # Gửi dữ liệu int16
        self._file = None
        self._queue = asyncio.Queue(maxsize=5) # Giới hạn queue để tránh đọc quá nhanh
        self._reader_task: asyncio.Task | None = None
        self._stopped = asyncio.Event()
        self._pts = 0 # Presentation timestamp counter
        self._time_base = f"1/{target_sample_rate}"

    async def start(self):
        """Mở file và bắt đầu task đọc file."""
        if self._reader_task is None:
            logger.info("Starting audio file reader task...")
            try:
                self._file = sf.SoundFile(self.file_path, 'r')
                logger.info(f"Opened sound file: {self.file_path} (SR={self._file.samplerate}, CH={self._file.channels})")
                if self._file.samplerate != self.target_sample_rate or self._file.channels != self.target_channels:
                     logger.warning(f"File gốc có SR={self._file.samplerate}, Ch={self._file.channels}. "
                                    f"Sẽ cố gắng đọc và chuyển đổi sang SR={self.target_sample_rate}, Ch={self.target_channels}")
                self._reader_task = asyncio.create_task(self._read_chunks())
            except Exception as e:
                 logger.exception(f"Failed to open sound file or start reader task: {e}")
                 self._stopped.set() # Dừng nếu không mở được file
                 raise
        else:
             logger.warning("Reader task already started.")

    def stop(self):
        """Dừng track và đóng file (phương thức đồng bộ)."""
        logger.info("Stopping AudioChunkTrack (sync call)...")
        self._stopped.set() # Báo hiệu dừng cho vòng lặp đọc và recv
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            # Không await task ở đây
        if self._file and not self._file.closed:
            try:
                self._file.close()
                logger.debug("Sound file closed in stop().")
            except Exception as e:
                 logger.warning(f"Exception closing file in stop(): {e}")
            self._file = None
        logger.info("AudioChunkTrack stop() finished (sync).")


    async def _read_chunks(self):
        """Đọc file audio thành các chunk và đưa vào queue."""
        logger.info("Audio reader task started execution.")
        chunk_count = 0
        try:
            while not self._stopped.is_set():
                # Đọc một chunk từ file
                data_float = self._file.read(frames=self.chunk_samples, dtype='float64', always_2d=False)

                if data_float is None or len(data_float) == 0:
                    logger.info("End of audio file reached.")
                    await self._queue.put(None) # Gửi tín hiệu kết thúc
                    break

                chunk_count += 1
                logger.debug(f"Read chunk {chunk_count} ({len(data_float)} samples float64)")

                # --- Xử lý Channels và Sample Rate (Đơn giản hóa) ---
                processed_data_float = data_float # Bắt đầu với dữ liệu gốc
                current_channels = self._file.channels
                current_samplerate = self._file.samplerate

                # Chuyển kênh (nếu cần)
                if current_channels != self.target_channels:
                     if current_channels > 1 and self.target_channels == 1:
                         processed_data_float = np.mean(processed_data_float.reshape(-1, current_channels), axis=1)
                         logger.debug(f"Mixed down channels from {current_channels} to {self.target_channels}")
                     else:
                         logger.warning(f"Channel conversion from {current_channels} to {self.target_channels} not fully implemented, might be incorrect.")
                         if len(processed_data_float.shape) > 1:
                              processed_data_float = processed_data_float[:, 0]

                # Resample (nếu cần)
                if current_samplerate != self.target_sample_rate:
                    logger.debug(f"Resampling from {current_samplerate}Hz to {self.target_sample_rate}Hz...")
                    num_samples_in = len(processed_data_float)
                    num_samples_out = int(num_samples_in * self.target_sample_rate / current_samplerate)
                    if num_samples_in > 0 and num_samples_out > 0:
                        # Tạo index thời gian cho dữ liệu vào và ra
                        time_in = np.linspace(0., float(num_samples_in) / current_samplerate, num_samples_in)
                        time_out = np.linspace(0., float(num_samples_in) / current_samplerate, num_samples_out)
                        processed_data_float = np.interp(time_out, time_in, processed_data_float)
                        logger.debug(f"Resampled to {len(processed_data_float)} samples.")
                    else:
                        processed_data_float = np.array([], dtype=np.float64)
                        logger.debug("Resampling resulted in empty array.")

                # Chuyển đổi sang int16
                if len(processed_data_float) > 0:
                    # Chuẩn hóa trước khi chuyển đổi để tránh clipping nếu giá trị float > 1.0
                    max_val = np.max(np.abs(processed_data_float))
                    if max_val > 1.0:
                        logger.warning(f"Audio chunk has values > 1.0 (max: {max_val}), normalizing before int16 conversion.")
                        processed_data_float = processed_data_float / max_val

                    data_int16 = (processed_data_float * 32767.0).astype(self.dtype)
                    await self._queue.put(data_int16)
                    logger.debug(f"Queued audio chunk {chunk_count} with {len(data_int16)} samples (int16).")
                else:
                    logger.debug(f"Empty audio chunk {chunk_count} after processing.")

                # Giả lập khoảng dừng
                pause_duration = random.uniform(MIN_PAUSE_S, MAX_PAUSE_S)
                logger.info(f"Pausing for {pause_duration:.2f} seconds after chunk {chunk_count}...")
                await asyncio.sleep(pause_duration)

        except sf.SoundFileError as e:
            logger.error(f"Error reading sound file: {e}")
            await self._queue.put(None)
        except asyncio.CancelledError:
             logger.info("Audio reader task cancelled.")
        except Exception as e:
            logger.exception(f"Unexpected error in audio reader task: {e}")
            await self._queue.put(None)
        finally:
            logger.info(f"Audio reader task finished after processing {chunk_count} chunks.")
            if self._file and not self._file.closed:
                self._file.close()


    async def recv(self) -> AudioFrame:
        """Lấy chunk audio từ queue và tạo AudioFrame."""
        if self._stopped.is_set():
            logger.debug("recv called on stopped track.")
            raise MediaStreamError("Track has been stopped.")
        if self._reader_task is None:
             raise MediaStreamError("Track not started. Call start() first.")

        # Đợi chunk tiếp theo từ queue
        # Thêm timeout để tránh bị kẹt nếu reader task gặp vấn đề âm thầm
        try:
             chunk_data = await asyncio.wait_for(self._queue.get(), timeout=MAX_PAUSE_S + CHUNK_DURATION_S + 5.0)
        except asyncio.TimeoutError:
             logger.error("Timeout waiting for audio chunk from queue. Reader task might be stuck.")
             await self.stop()
             raise MediaStreamError("Timeout receiving audio chunk.")

        if chunk_data is None: # Tín hiệu kết thúc từ reader
            logger.info("recv: Received None sentinel, stopping track.")
            await self.stop() # Dừng track một cách an toàn
            raise MediaStreamError("Audio stream finished.")
        elif self._stopped.is_set(): # Kiểm tra lại nếu bị dừng trong lúc chờ queue
             logger.debug("recv: Track stopped while waiting for queue.")
             raise MediaStreamError("Track has been stopped.")


        # Tạo AudioFrame
        samples_in_chunk = len(chunk_data) // self.target_channels
        if samples_in_chunk == 0:
             logger.warning("Received empty audio chunk data after queue.")
             # Có thể trả về frame trống hoặc raise lỗi tùy logic
             # Tạm thời raise lỗi để dễ thấy vấn đề
             raise MediaStreamError("Received empty audio chunk data.")

        # Lấy itemsize từ kiểu numpy một cách đúng đắn
        audio_format = f"s{np.dtype(self.dtype).itemsize * 8}" # Ví dụ: s16

        frame = AudioFrame(format=audio_format,
                           layout="mono" if self.target_channels == 1 else "stereo",
                           samples=samples_in_chunk)
        try:
             frame.planes[0].update(chunk_data.tobytes())
        except Exception as e:
             logger.exception(f"Error updating AudioFrame plane: {e}")
             raise MediaStreamError("Error creating audio frame") from e

        frame.sample_rate = self.target_sample_rate
        frame.time_base = self._time_base
        frame.pts = self._pts
        self._pts += samples_in_chunk # Cập nhật pts cho frame tiếp theo

        logger.debug(f"recv: Returning AudioFrame pts={frame.pts} samples={frame.samples}")
        return frame

# --- Logic Test Chính ---
async def run_test():
    global current_session_id
    pc = RTCPeerConnection(configuration=RTC_CONFIG)
    # Tạo data channel NGAY LẬP TỨC để sẵn sàng nhận sự kiện open
    data_channel = pc.createDataChannel("events")
    logger.debug("Data channel created (state: %s)", data_channel.readyState)

    audio_track: AudioChunkTrack | None = None # Khởi tạo là None

    received_events = asyncio.Queue()
    client_tasks = set()
    full_transcript = ""

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track {track.kind} received")
        @track.on("ended")
        async def on_ended():
            logger.info(f"Track {track.kind} (từ server) ended")

    @pc.on("datachannel")
    def on_datachannel(channel):
        # Logic này thường không xảy ra khi client tạo channel
        logger.warning(f"Data channel '{channel.label}' created by remote (state: {channel.readyState})")
        # Gán lại data_channel nếu server tạo và client chưa tạo? (Hiếm)
        # nonlocal data_channel
        # data_channel = channel
        # Cần đăng ký lại các handler on("open"), on("close"), on("message")

    @data_channel.on("open")
    def on_channel_open():
        logger.info(f"Data channel '{data_channel.label}' opened")
        # Gửi cấu hình ban đầu NGAY KHI KÊNH MỞ
        task = asyncio.create_task(send_initial_config(data_channel))
        client_tasks.add(task)
        task.add_done_callback(client_tasks.discard)

    @data_channel.on("close")
    def on_channel_close():
        logger.info(f"Data channel '{data_channel.label}' closed")

    @data_channel.on("message")
    async def on_channel_message(message):
        nonlocal full_transcript
        global current_session_id
        try:
            event_data = json.loads(message)
            validated_event = server_event_type_adapter.validate_python(event_data)
            logger.info(f"<<< Received event: {validated_event.type}")

            if isinstance(validated_event, SessionCreatedEvent):
                 current_session_id = validated_event.session.id
                 logger.info(f"    Captured Session ID: {current_session_id}")

            # --- XỬ LÝ DELTA CHO LIVE TRANSCRIPT ---
            transcript_changed = False
            if isinstance(validated_event, (ResponseTextDeltaEvent, ResponseAudioTranscriptDeltaEvent)):
                delta = validated_event.delta
                logger.info(f"    Delta: '{delta}'")
                full_transcript += delta
                transcript_changed = True
            elif isinstance(validated_event, ConversationItemInputAudioTranscriptionCompletedEvent):
                 logger.info(f"    Transcription COMPLETED for item {validated_event.item_id}")
                 final = validated_event.transcript
                 logger.info(f"    Final transcript for item: '{final}'")
                 if final is not None: # Đảm bảo final không phải None
                     full_transcript = final
                     transcript_changed = True
            elif isinstance(validated_event, ErrorEvent):
                 logger.error(f"    SERVER ERROR: {validated_event.error.message} (Type: {validated_event.error.type}, Code: {validated_event.error.code})")

            if transcript_changed:
                 print(f"\rTranscript: {full_transcript}   ", end="", flush=True) # Thêm khoảng trắng để xóa ký tự thừa

            await received_events.put(validated_event)
        except json.JSONDecodeError:
            logger.error(f"    Failed to decode JSON: {message}")
        except Exception as e:
            logger.exception(f"    Failed to validate server event: {e}") # Dùng exception để log cả traceback

    # --- Bắt đầu kết nối ---
    try:
        # >>> TẠO TRACK TÙY CHỈNH <<<
        audio_track = AudioChunkTrack(AUDIO_FILE_PATH, CHUNK_DURATION_S, TARGET_SAMPLE_RATE, TARGET_CHANNELS)
        await audio_track.start() # Bắt đầu đọc file
        pc.addTrack(audio_track)

        await pc.setLocalDescription(await pc.createOffer())
        offer_sdp = pc.localDescription.sdp
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
        logger.info("Created Offer SDP")

        headers = {"Content-Type": "text/plain"}
        params = {"model": MODEL_ID}
        logger.info(f"Sending POST request to {API_ENDPOINT}")
        async with aiohttp.ClientSession() as session:
             async with session.post(API_ENDPOINT, params=params, data=offer_sdp, headers=headers, ssl=False) as response:
                 response.raise_for_status()
                 answer_sdp = await response.text()
                 logger.info(f"Received Answer SDP (status: {response.status})")

        answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
        await pc.setRemoteDescription(answer)
        logger.info("Set Remote Description")

        await wait_for_connection(pc)

        # Chờ data channel mở (có thể đã mở do handler ở trên)
        if data_channel.readyState != "open":
            logger.info("Waiting for data channel to open...")
            # Thêm timeout chờ kênh mở
            try:
                await asyncio.wait_for(data_channel.transport.transport.sctp.wait_established(), timeout=10.0)
                # Chờ thêm chút để on("open") được gọi nếu chưa
                await asyncio.sleep(0.1)
                if data_channel.readyState != "open":
                     raise ConnectionError("Data channel did not open.")
                logger.info("Data channel reached 'open' state after wait.")
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for Data Channel to open.")
                raise ConnectionError("Data channel open timed out.")

        logger.info("Streaming audio chunks from file with pauses...")

        # --- Chờ cho đến khi track audio kết thúc gửi ---
        if audio_track and audio_track._reader_task:
             logger.info("Waiting for audio reader task to complete...")
             await audio_track._reader_task # Chờ task đọc file xong
             logger.info("Audio reader task completed.")
        else:
             logger.warning("Audio reader task was not started correctly or track is missing.")

        logger.info("Audio file streaming finished.")
        logger.info("Waiting a bit longer for final server events...")
        # Giảm thời gian chờ vì VAD server nên xử lý nhanh hơn
        await asyncio.sleep(3.0) # Chờ 3 giây

    except Exception as e:
         logger.exception(f"An error occurred during connection or streaming: {e}")
    finally:
         # --- Dọn dẹp ---
         logger.info("\n" + "-" * 30) # Xuống dòng sau khi in live transcript
         logger.info("Final Full transcript received:")
         logger.info(full_transcript if full_transcript else "[No transcript received]")
         logger.info("-" * 30)

         logger.info("Closing connection...")
         # Hủy các task phụ trợ trước
         for task in client_tasks:
             if not task.done():
                 task.cancel()
         try:
             await asyncio.gather(*client_tasks, return_exceptions=True)
         except asyncio.CancelledError:
             pass

         # Dừng audio track
         if audio_track:
             audio_track.stop()

         # Đóng PeerConnection
         if pc and pc.connectionState != "closed":
             await pc.close()
             logger.info("PeerConnection closed.")
         else:
              logger.info("PeerConnection already closed or not initialized.")

         logger.info("Connection closed routine finished.")

async def send_initial_config(channel: RTCDataChannel):
    logger.info("Sending initial configuration...")
    try:
        # Gửi cấu hình ban đầu - ví dụ: TẮT VAD nếu server hỗ trợ
        # Bạn có thể bỏ qua phần này nếu muốn dùng VAD server mặc định
        update_data = PartialSession(
             turn_detection=None, # Yêu cầu tắt VAD
        )
        # Chỉ gửi nếu thực sự có cấu hình cần update (ví dụ: turn_detection=None)
        fields_to_send = {'turn_detection': None} # Ví dụ: chỉ gửi turn_detection
        if fields_to_send:
            # Đảm bảo truyền đúng kiểu vào PartialSession
            session_payload = PartialSession(turn_detection=None) # Chỉ gửi turn_detection=None
            session_update = SessionUpdateEvent(type="session.update", session=session_payload)
            logger.info(f">>> Sending: {session_update.type} with data: turn_detection=None")
            channel.send(session_update.model_dump_json())
            await asyncio.sleep(0.1)
        else:
             logger.info("No specific initial config fields set to send.")

        # Gửi ResponseCreateEvent để server biết bắt đầu xử lý/mong đợi phản hồi
        create_response = ResponseCreateEvent(type="response.create")
        logger.info(f">>> Sending: {create_response.type}")
        channel.send(create_response.model_dump_json())

        logger.info("Finished sending initial configuration.")
    except Exception as e:
        logger.exception(f"Error sending initial configuration: {e}") # Dùng logger.exception

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user.")
    except Exception as e:
         # Log lỗi cuối cùng nếu có exception không được bắt trong run_test
         logger.exception(f"Unhandled error during test run: {e}")
    finally:
         print() # Đảm bảo con trỏ xuống dòng