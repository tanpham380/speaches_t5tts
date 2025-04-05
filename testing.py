import asyncio
import json
import logging
import time
import ssl
import os
import random # Để tạo khoảng dừng ngẫu nhiên (tùy chọn)
from fractions import Fraction # <<< THÊM IMPORT FRACTION

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
# --- Bật Log DEBUG cho aiortc (tùy chọn) ---
# logging.getLogger("aiortc").setLevel(logging.DEBUG)
# logging.getLogger("aioice").setLevel(logging.DEBUG)
# logging.getLogger("av").setLevel(logging.DEBUG)

SERVER_URL = "http://localhost:8000" # Thay đổi nếu cần
API_ENDPOINT = f"{SERVER_URL}/v1/realtime"
MODEL_ID = "Systran/faster-whisper-large-v3" # Model STT bạn muốn test
AUDIO_FILE_PATH = "generated_429000_long.wav" # File audio nguồn

# --- Cấu hình gửi Chunk ---
CHUNK_DURATION_S = 0.04  # Gửi chunk 40ms (phù hợp hơn cho Opus)
# MIN_PAUSE_S = 0.5     # Không cần pause nếu stream liên tục
# MAX_PAUSE_S = 2.0
TARGET_SAMPLE_RATE = 48000 # Sample rate gửi qua WebRTC
TARGET_CHANNELS = 1        # Gửi kênh mono

# Kiểm tra file tồn tại
if not os.path.exists(AUDIO_FILE_PATH):
    logger.error(f"File âm thanh không tìm thấy tại: {AUDIO_FILE_PATH}")
    exit(1)

# Cấu hình WebRTC
ICE_SERVERS = []
RTC_CONFIG = RTCConfiguration(iceServers=[RTCIceServer(**s) for s in ICE_SERVERS])

# Biến toàn cục để lưu session ID
current_session_id = None

# --- Hàm trợ giúp ---
async def wait_for_connection(pc: RTCPeerConnection):
    """Chờ cho đến khi kết nối ICE thành công."""
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE Connection State is {pc.iceConnectionState}")
        # Không raise lỗi trực tiếp ở đây

    t_start = time.time()
    timeout = 20.0
    while time.time() - t_start < timeout:
        state = pc.iceConnectionState
        if state in ["completed", "connected"]:
            logger.info("ICE Connection Established.")
            return
        if state in ["failed", "closed", "disconnected"]:
             logger.error(f"ICE Connection failed or closed prematurely ({state})")
             raise ConnectionError(f"ICE Connection failed or closed prematurely ({state})")
        await asyncio.sleep(0.1)

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
        self.dtype = np.int16
        self._file = None
        self._queue = asyncio.Queue(maxsize=10) # Tăng queue size một chút
        self._reader_task: asyncio.Task | None = None
        self._stopped = asyncio.Event()
        self._pts = 0
        try:
            self._time_base = Fraction(1, target_sample_rate) # <<< SỬA DÙNG FRACTION
        except ZeroDivisionError:
             raise ValueError("Invalid target sample rate")
        except ValueError:
             raise ValueError("Invalid time base")

    async def start(self):
        """Mở file và bắt đầu task đọc file."""
        if self._reader_task is None:
            logger.info("Starting audio file reader task...")
            try:
                self._file = sf.SoundFile(self.file_path, 'r')
                logger.info(f"Opened sound file: {self.file_path} (SR={self._file.samplerate}, CH={self._file.channels})")
                if self._file.samplerate != self.target_sample_rate or self._file.channels != self.target_channels:
                     logger.warning(f"File gốc SR/Ch khác target. Sẽ đọc và chuyển đổi.")
                self._reader_task = asyncio.create_task(self._read_chunks())
            except Exception as e:
                 logger.exception(f"Failed to open sound file or start reader task: {e}")
                 self._stopped.set()
                 raise
        else:
             logger.warning("Reader task already started.")

    def stop(self):
        """Dừng track và đóng file (đồng bộ)."""
        logger.info("Stopping AudioChunkTrack (sync call)...")
        self._stopped.set()
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
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
        processed_samples = 0
        silence_chunk_samples = int(1.0 * self.target_sample_rate)
        silence_chunk_data = np.zeros(silence_chunk_samples, dtype=self.dtype)

        try:
            while not self._stopped.is_set():
                # Đọc chính xác số sample cho chunk duration ở sample rate gốc
                read_samples = int(CHUNK_DURATION_S * self._file.samplerate)
                data_float = self._file.read(frames=read_samples, dtype='float64', always_2d=False)

                if data_float is None or len(data_float) == 0:
                    logger.info("End of audio file reached.")
                    await self._queue.put(None)
                    break


                chunk_count += 1
                logger.debug(f"Read chunk {chunk_count} ({len(data_float)} samples float64 at {self._file.samplerate}Hz)")

                # --- Xử lý Channels và Sample Rate ---
                processed_data_float = data_float
                current_channels = self._file.channels
                current_samplerate = self._file.samplerate

                if current_channels != self.target_channels:
                     if current_channels > 1 and self.target_channels == 1:
                         processed_data_float = np.mean(processed_data_float.reshape(-1, current_channels), axis=1)
                     else: # Các trường hợp khác phức tạp hơn
                         logger.warning(f"Channel conversion from {current_channels} to {self.target_channels} needs better implementation.")
                         if len(processed_data_float.shape) > 1: processed_data_float = processed_data_float[:, 0]

                if current_samplerate != self.target_sample_rate:
                    num_samples_in = len(processed_data_float)
                    num_samples_out = int(num_samples_in * self.target_sample_rate / current_samplerate)
                    if num_samples_in > 0 and num_samples_out > 0:
                        time_in = np.linspace(0., float(num_samples_in) / current_samplerate, num_samples_in, endpoint=False)
                        time_out = np.linspace(0., float(num_samples_in) / current_samplerate, num_samples_out, endpoint=False)
                        processed_data_float = np.interp(time_out, time_in, processed_data_float)
                    else:
                        processed_data_float = np.array([], dtype=np.float64)

                # Chuyển đổi sang int16
                if len(processed_data_float) > 0:
                    max_val = np.max(np.abs(processed_data_float))
                    if max_val > 1.0: processed_data_float = processed_data_float / max_val
                    data_int16 = (processed_data_float * 32767.0).astype(self.dtype)

                    # Kiểm tra xem số sample có khớp với chunk_samples mong đợi ở target rate không
                    expected_samples = int(CHUNK_DURATION_S * self.target_sample_rate)
                    if len(data_int16) != expected_samples:
                        logger.warning(f"Chunk {chunk_count} sample count mismatch after processing: expected {expected_samples}, got {len(data_int16)}. Padding/truncating.")
                        # Pad hoặc truncate để khớp (padding bằng 0)
                        if len(data_int16) < expected_samples:
                             padding = np.zeros(expected_samples - len(data_int16), dtype=self.dtype)
                             data_int16 = np.concatenate((data_int16, padding))
                        else:
                             data_int16 = data_int16[:expected_samples]

                    await self._queue.put(data_int16)
                    processed_samples += len(data_int16)



                    logger.debug(f"Queued audio chunk {chunk_count} ({len(data_int16)} samples int16 at {self.target_sample_rate}Hz). Total processed: {processed_samples}")
                else:
                    logger.debug(f"Empty audio chunk {chunk_count} after processing.")
                if chunk_count > 0 and chunk_count % 5 == 0: # Ví dụ: chèn sau mỗi 5 chunk
                    logger.info(f"Injecting silence chunk ({silence_chunk_samples} samples)...")
                    await self._queue.put(silence_chunk_data)
                    # Không cần sleep ở đây nếu VAD server xử lý

                if self._queue.full():
                    logger.warning("Audio queue is full, reader sleeping briefly...")
                    await asyncio.sleep(CHUNK_DURATION_S / 4) # Ngủ chút nếu queue đầy

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

        try:
             # Tăng timeout một chút đề phòng reader bị chậm
             chunk_data = await asyncio.wait_for(self._queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
             logger.error("Timeout waiting for audio chunk from queue.")
             self.stop() # Dừng track nếu không nhận được dữ liệu
             raise MediaStreamError("Timeout receiving audio chunk.")

        if chunk_data is None: # Tín hiệu kết thúc
            logger.info("recv: Received None sentinel, stopping track.")
            self.stop()
            raise MediaStreamError("Audio stream finished.")
        elif self._stopped.is_set():
             logger.debug("recv: Track stopped while waiting for queue.")
             raise MediaStreamError("Track has been stopped.")

        samples_in_chunk = len(chunk_data) // self.target_channels
        if samples_in_chunk == 0:
             logger.warning("recv: Received empty audio chunk data.")
             # Trả về frame trống thay vì lỗi để tránh làm sập sender
             frame = AudioFrame(format="s16", layout="mono", samples=0)
             frame.sample_rate = self.target_sample_rate
             frame.time_base = self._time_base
             frame.pts = self._pts
             # Không tăng pts cho frame trống
             return frame
             # raise MediaStreamError("Received empty audio chunk data.")

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
        self._pts += samples_in_chunk # Cập nhật pts

        logger.debug(f"recv: Returning AudioFrame pts={frame.pts} samples={frame.samples}")
        return frame

# --- Logic Test Chính ---
async def run_test():
    global current_session_id
    pc = RTCPeerConnection(configuration=RTC_CONFIG)
    data_channel = pc.createDataChannel("events")
    audio_track: AudioChunkTrack | None = None
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
        logger.warning(f"Data channel '{channel.label}' created by remote (state: {channel.readyState})")

    @data_channel.on("open")
    def on_channel_open():
        logger.info(f"Data channel '{data_channel.label}' opened")
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
                 if final is not None:
                     full_transcript = final
                     transcript_changed = True
            elif isinstance(validated_event, ErrorEvent):
                 logger.error(f"    SERVER ERROR: {validated_event.error.message} (Type: {validated_event.error.type}, Code: {validated_event.error.code})")

            if transcript_changed:
                 print(f"\rTranscript: {full_transcript}   ", end="", flush=True)

            await received_events.put(validated_event)
        except json.JSONDecodeError:
            logger.error(f"    Failed to decode JSON: {message}")
        except Exception as e:
            logger.exception(f"    Failed to validate server event: {e}")

    # --- Bắt đầu kết nối ---
    try:
        audio_track = AudioChunkTrack(AUDIO_FILE_PATH, CHUNK_DURATION_S, TARGET_SAMPLE_RATE, TARGET_CHANNELS)
        await audio_track.start()
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

        if data_channel.readyState != "open":
            logger.info("Waiting for data channel to open...")
            try:
                await asyncio.wait_for(data_channel.transport.transport.sctp.wait_established(), timeout=10.0)
                await asyncio.sleep(0.1)
                if data_channel.readyState != "open": raise ConnectionError("Data channel did not open.")
                logger.info("Data channel reached 'open' state after wait.")
            except asyncio.TimeoutError:
                raise ConnectionError("Data channel open timed out.")

        logger.info("Streaming audio chunks from file...")

        # --- Chờ task đọc file hoàn thành ---
        if audio_track and audio_track._reader_task:
             logger.info("Waiting for audio reader task to complete...")
             # Chờ task đọc file, không cần timeout ở đây vì nó sẽ tự kết thúc
             await audio_track._reader_task
             logger.info("Audio reader task completed.")
        else:
             logger.warning("Audio reader task was not started correctly or track is missing.")

        logger.info("Audio file streaming finished.")
        logger.info("Waiting a bit longer for final server events...")
        await asyncio.sleep(10.0)

    except Exception as e:
         logger.exception(f"An error occurred during connection or streaming: {e}")
    finally:
         # --- Dọn dẹp ---
         print() # Xuống dòng sau live transcript
         logger.info("\n" + "-" * 30)
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

         # Dừng audio track nếu tồn tại
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
        # --- COMMENT HOẶC XÓA PHẦN UPDATE VAD NÀY ---
        # update_data = PartialSession(turn_detection=None)
        # fields_to_send = {'turn_detection': None}
        # if fields_to_send:
        #     session_payload = PartialSession(turn_detection=None)
        #     session_update = SessionUpdateEvent(type="session.update", session=session_payload)
        #     logger.info(f">>> Sending: {session_update.type} to disable VAD (if supported)")
        #     channel.send(session_update.model_dump_json())
        #     await asyncio.sleep(0.1)
        # else:
        #      logger.info("No specific initial config fields set to send.")
        # --- KẾT THÚC COMMENT/XÓA ---

        # Vẫn gửi ResponseCreateEvent
        create_response = ResponseCreateEvent(type="response.create")
        logger.info(f">>> Sending: {create_response.type}")
        channel.send(create_response.model_dump_json())

        logger.info("Finished sending initial configuration.")
    except Exception as e:
        logger.exception(f"Error sending initial configuration: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user.")
    except Exception as e:
         logger.exception(f"Unhandled error during test run: {e}")
    finally:
         print() # Đảm bảo con trỏ xuống dòng