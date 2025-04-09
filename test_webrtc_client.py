import asyncio
import logging
import time
import json
import numpy as np
import argparse
from pathlib import Path
import io

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    MediaStreamTrack,
    RTCDataChannel # Đảm bảo import RTCDataChannel
)
from aiortc.contrib.media import MediaBlackhole
from av import AudioFrame
import httpx
from pydub import AudioSegment
from pydub.utils import make_chunks

# --- Thêm Import Event từ mouble_types ---
try:
    from mouble_types.realtime import (
        SessionUpdateEvent,
        ResponseCreateEvent,
        PartialSession,
        NotGiven,
        NOT_GIVEN,
        InputAudioBufferCommitEvent, # Cần cho commit
        SessionCreatedEvent,         # Cần để lấy session_id
        server_event_type_adapter,   # Cần để parse server events
        # client_event_type_adapter, # Ít cần hơn nếu chỉ gửi event cụ thể
        ErrorEvent,                  # Để log lỗi server nếu có
        ResponseTextDeltaEvent,      # Để log transcript
        ResponseAudioTranscriptDeltaEvent, # Để log transcript
        ConversationItemInputAudioTranscriptionCompletedEvent, # Để log transcript
    )
except ImportError as e:
    print(f"Lỗi import mouble_types.realtime: {e}")
    print("Vui lòng đảm bảo file mouble_types/realtime.py có thể truy cập được.")
    exit(1)
# --- Kết thúc thêm Import ---


# --- Configuration ---
SEND_AUDIO_ENABLED = True
RECEIVE_AUDIO_ENABLED = True
RTC_SEND_SAMPLE_RATE = 48000 # Giữ nguyên như phiên bản FileAudioTrack cũ
RTC_SEND_CHANNELS = 2
RTC_SEND_FORMAT_AV = "s16"
RTC_SEND_LAYOUT_AV = "stereo"
RTC_SEND_SAMPLE_WIDTH = 2
RTC_SEND_DTYPE = np.int16
AUDIO_CHUNK_DURATION_MS = 1000
AUDIO_PTIME = AUDIO_CHUNK_DURATION_MS / 1000.0
SAMPLES_PER_FRAME = int(RTC_SEND_SAMPLE_RATE * AUDIO_PTIME)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebRTCClientTest_FileTrack") # Đổi tên logger
logging.getLogger("pydub").setLevel(logging.WARNING)
# logging.getLogger("aiortc").setLevel(logging.DEBUG)
# logging.getLogger("aioice").setLevel(logging.DEBUG)

# --- Global Event và Session ID ---
session_created_event = asyncio.Event()
current_session_id = None
# --- Kết thúc Global Event và Session ID ---

# --- Helper: Audio Processing (stream_audio_chunks - không đổi) ---
async def stream_audio_chunks(
    file_path: str,
    chunk_duration_ms: int,
    target_sample_rate: int,
    target_channels: int,
    target_sample_width: int,
    target_format: str = 's16le'
):
    logger.info(f"Loading audio file: {file_path}")
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise

    logger.info(
        f"Original audio: {audio.frame_rate}Hz, {audio.channels}ch, "
        f"{audio.sample_width * 8}-bit, Duration: {audio.duration_seconds:.2f}s"
    )

    resample_needed = False
    if audio.frame_rate != target_sample_rate:
        logger.info(f"Resampling rate from {audio.frame_rate}Hz to {target_sample_rate}Hz.")
        audio = audio.set_frame_rate(target_sample_rate)
        resample_needed = True
    if audio.channels != target_channels:
        logger.info(f"Setting channels from {audio.channels} to {target_channels}.")
        audio = audio.set_channels(target_channels)
        resample_needed = True
    if audio.sample_width != target_sample_width:
         logger.info(f"Target sample width ({target_sample_width*8}-bit) differs from source ({audio.sample_width*8}-bit). Export will handle conversion.")
         resample_needed = True

    if resample_needed:
        logger.info(f"Audio prepared for streaming: {target_sample_rate}Hz, {target_channels}ch, {target_sample_width*8}-bit")

    chunks = make_chunks(audio, chunk_duration_ms)
    total_chunks = len(chunks)

    logger.info(f"Ready to stream {total_chunks} chunks of {chunk_duration_ms}ms each (waiting for session)...")

    logger.info("stream_audio_chunks: Waiting for session_created_event...")
    await session_created_event.wait()
    logger.info("stream_audio_chunks: session_created_event received! Starting audio stream.")
    start_time = time.perf_counter()
    playback_time_offset = 0.0

    for i, chunk in enumerate(chunks):
        buffer = io.BytesIO()
        try:
            chunk_to_export = chunk.set_frame_rate(target_sample_rate).set_channels(target_channels)
            chunk_to_export.export(buffer, format=target_format)
            raw_bytes = buffer.getvalue()
        except Exception as e:
            logger.error(f"Error exporting chunk {i+1} to format '{target_format}': {e}. Skipping chunk.")
            continue

        if not raw_bytes:
            logger.debug(f"Skipping empty chunk {i+1} after export.")
            continue

        bytes_per_frame = target_channels * target_sample_width
        if len(raw_bytes) % bytes_per_frame != 0:
             num_frames = len(raw_bytes) // bytes_per_frame
             expected_bytes = num_frames * bytes_per_frame
             logger.warning(
                 f"Chunk {i+1} data length {len(raw_bytes)} is not perfectly divisible by frame size {bytes_per_frame}. "
                 f"Truncating to {expected_bytes} bytes ({num_frames} frames)."
             )
             raw_bytes = raw_bytes[:expected_bytes]
             if not raw_bytes:
                 logger.debug(f"Chunk {i+1} became empty after truncation. Skipping.")
                 continue

        yield raw_bytes

        playback_time_offset += chunk_duration_ms / 1000.0
        expected_playback_time = start_time + playback_time_offset
        current_time = time.perf_counter()
        wait_time = expected_playback_time - current_time
        if wait_time > 0:
            await asyncio.sleep(wait_time)

    logger.info("Finished yielding all audio chunks from file.")


# --- File Audio Track to Send (recv - không đổi) ---
class FileAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        self._generator = stream_audio_chunks(
            file_path=str(self.file_path),
            chunk_duration_ms=AUDIO_CHUNK_DURATION_MS,
            target_sample_rate=RTC_SEND_SAMPLE_RATE,
            target_channels=RTC_SEND_CHANNELS,
            target_sample_width=RTC_SEND_SAMPLE_WIDTH,
            target_format='s16le'
        )
        self._frame_pts = 0
        self._first_frame = True
        self._start_time = None
        self._stopped = asyncio.Event()

    async def recv(self) -> AudioFrame:
        await asyncio.sleep(AUDIO_PTIME / 10) # << Rất nhỏ sleep để tránh CPU hogging
        if self._stopped.is_set():
             raise StopAsyncIteration

        try:
            chunk_bytes = await self._generator.__anext__()
            if self._first_frame:
                logger.info("FileAudioTrack: Received first audio chunk from generator.")
                self._start_time = time.time()
                self._first_frame = False

        except StopAsyncIteration:
            logger.info("File audio stream finished (generator stopped).")
            self._stopped.set()
            raise StopAsyncIteration
        except Exception as e:
             logger.error(f"Error getting audio chunk from generator: {e}", exc_info=True)
             self._stopped.set()
             raise StopAsyncIteration

        try:
            samples_in_chunk = len(chunk_bytes) // (RTC_SEND_CHANNELS * RTC_SEND_SAMPLE_WIDTH)
            if samples_in_chunk == 0:
                 logger.warning("Received empty chunk_bytes after generator yield, skipping frame.")
                 return await self.recv()

            samples_np = np.frombuffer(chunk_bytes, dtype=RTC_SEND_DTYPE)
            # Đảm bảo đúng shape cho stereo s16
            if samples_np.size != samples_in_chunk * RTC_SEND_CHANNELS:
                 logger.error(f"Buffer size mismatch: expected {samples_in_chunk * RTC_SEND_CHANNELS}, got {samples_np.size}")
                 # Có thể cần xử lý lỗi ở đây, ví dụ skip frame
                 return await self.recv()
            try:
                 samples_reshaped = np.ascontiguousarray(samples_np.reshape((samples_in_chunk, RTC_SEND_CHANNELS)))
            except ValueError as reshape_err:
                 logger.error(f"Reshape error: {reshape_err}. samples_in_chunk={samples_in_chunk}, RTC_SEND_CHANNELS={RTC_SEND_CHANNELS}, samples_np.size={samples_np.size}")
                 return await self.recv()


            frame = AudioFrame(format=RTC_SEND_FORMAT_AV, layout=RTC_SEND_LAYOUT_AV, samples=samples_in_chunk)
            frame.sample_rate = RTC_SEND_SAMPLE_RATE
            frame.pts = self._frame_pts
            # frame.time_base = '1/48000' # Quan trọng: Đặt time_base
            from fractions import Fraction
            frame.time_base = Fraction(1, RTC_SEND_SAMPLE_RATE)


            # logger.debug(f"Creating frame: pts={frame.pts}, samples={frame.samples}, time={frame.time:.4f}")

            if len(frame.planes) > 0:
                 expected_bytes = samples_reshaped.nbytes
                 if frame.planes[0].buffer_size >= expected_bytes:
                     frame.planes[0].update(samples_reshaped.tobytes())
                 else:
                     logger.error(f"AudioFrame plane buffer size ({frame.planes[0].buffer_size}) "
                                  f"is smaller than numpy array bytes ({expected_bytes}). Skipping frame.")
                     return await self.recv()
            else:
                 logger.error("Cannot update AudioFrame: No planes available.")
                 raise ValueError("AudioFrame created without planes")

            self._frame_pts += samples_in_chunk
            return frame

        except ValueError as e:
             logger.error(f"ValueError during AudioFrame processing: {e}", exc_info=True)
             self._stopped.set()
             raise StopAsyncIteration
        except Exception as e:
             logger.error(f"Error processing audio chunk into frame: {e}", exc_info=True)
             self._stopped.set()
             raise StopAsyncIteration

    async def stop(self):
        logger.debug("FileAudioTrack stop() called.")
        self._stopped.set()
        if hasattr(self._generator, 'aclose'):
             try:
                await self._generator.aclose()
             except GeneratorExit:
                 pass # Bỏ qua lỗi này khi đóng generator

# --- Hàm gửi cấu hình ban đầu ---
async def send_initial_config(channel: RTCDataChannel):
    logger.info("Sending initial configuration...")
    try:
        # Tùy chọn: Gửi SessionUpdateEvent (ví dụ: tắt VAD)
        # update_data = PartialSession(turn_detection=None)
        # fields_to_send = {k: v for k, v in update_data.model_dump().items() if not isinstance(v, NotGiven)}
        # if fields_to_send:
        #     session_update = SessionUpdateEvent(session=PartialSession(**fields_to_send))
        #     logger.info(f">>> Sending DC: {session_update.type} with data {fields_to_send}")
        #     channel.send(session_update.model_dump_json())
        # else:
        #      logger.info("No specific initial config fields to send (SessionUpdate).")

        # Luôn gửi ResponseCreateEvent để kích hoạt server
        create_response = ResponseCreateEvent(type="response.create")
        logger.info(f">>> Sending DC: {create_response.type}")
        channel.send(create_response.model_dump_json())

        await asyncio.sleep(0.5)
        logger.info("Finished sending initial configuration.")
    except Exception as e:
        logger.error(f"Error sending initial configuration via Data Channel: {e}")
# --- Kết thúc hàm gửi cấu hình ban đầu ---

# --- Main Test Function (run_test) ---
async def run_test(server_url: str, model_name: str, audio_file: str):
    global current_session_id # Cần để ghi vào
    logger.info("Starting WebRTC client test (using FileAudioTrack with Commit)...")
    session_created_event.clear()
    current_session_id = None # Reset session ID

    config = RTCConfiguration()
    pc = RTCPeerConnection(configuration=config)
    tasks = set()
    file_audio_track = None
    full_transcript = "" # Biến lưu transcript

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state is {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
             logger.error("ICE connection failed. Shutting down.")
             # Đảm bảo đóng kết nối và dừng task
             if file_audio_track and not file_audio_track._stopped.is_set():
                  asyncio.create_task(file_audio_track.stop())
             await pc.close() # Đóng pc ở đây

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed", "disconnected"]:
             logger.warning(f"Peer connection state changed to {pc.connectionState}. Stopping audio send if active.")
             if file_audio_track and not file_audio_track._stopped.is_set():
                 asyncio.create_task(file_audio_track.stop())


    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.warning(f"Data channel '{channel.label}' created by remote (unexpected)")

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track '{track.kind}' received from server")
        if track.kind == "audio" and RECEIVE_AUDIO_ENABLED:
            recorder = MediaBlackhole()
            recorder.addTrack(track)
            task = asyncio.create_task(recorder.start())
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            logger.info("Started consuming received audio track.")

            @track.on("ended")
            async def on_ended():
                logger.info("Server audio track ended")
                if recorder and hasattr(recorder,'stop'):
                    logger.info("Stopping audio recorder.")
                    try:
                       await recorder.stop()
                    except Exception as rec_stop_e:
                       logger.warning(f"Error stopping recorder: {rec_stop_e}")

        elif track.kind == "video":
             logger.warning("Received unexpected video track")

    # --- Create Data Channel ---
    # Đặt tên là "events" như client kia nếu cần thống nhất
    channel = pc.createDataChannel("events")
    logger.info(f"Created data channel '{channel.label}'")

    @channel.on("open")
    def on_open():
        logger.info(f"Data channel '{channel.label}' is open")
        task = asyncio.create_task(send_initial_config(channel))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    @channel.on("close")
    def on_close():
         logger.info(f"Data channel '{channel.label}' is closed")

    # --- Sửa on_message ---
    @channel.on("message")
    def on_message(message):
        nonlocal full_transcript # Cho phép ghi vào biến bên ngoài
        global current_session_id
        try:
            event_data = json.loads(message)
            validated_event = server_event_type_adapter.validate_python(event_data)
            event_type = validated_event.type
            logger.info(f"< Server DC ({event_type}): {json.dumps(event_data)}")

            if isinstance(validated_event, SessionCreatedEvent):
                current_session_id = validated_event.session.id
                logger.info(f">>> Received session.created! Captured Session ID: {current_session_id} <<<")
                session_created_event.set()

            # --- Xử lý các sự kiện phiên âm ---
            elif isinstance(validated_event, ResponseTextDeltaEvent):
                logger.info(f"    Text Delta: '{validated_event.delta}'")
                full_transcript += validated_event.delta
            elif isinstance(validated_event, ResponseAudioTranscriptDeltaEvent):
                 logger.info(f"    Audio Transcript Delta: '{validated_event.delta}'")
                 full_transcript += validated_event.delta
            elif isinstance(validated_event, ConversationItemInputAudioTranscriptionCompletedEvent):
                 logger.info(f"    Transcription COMPLETED for item {validated_event.item_id}")
                 logger.info(f"    Final transcript for item: '{validated_event.transcript}'")
                 # Có thể cập nhật full_transcript ở đây nếu cần bản cuối cùng
                 # full_transcript = validated_event.transcript
            elif isinstance(validated_event, ErrorEvent):
                 logger.error(f"    SERVER ERROR: {validated_event.error.message} (Type: {validated_event.error.type}, Code: {validated_event.error.code})")

        except json.JSONDecodeError:
            logger.warning(f"< Server DC non-JSON: {message}")
        except Exception as e:
             logger.error(f"Error processing server message: {e}", exc_info=True)
    # --- Kết thúc Sửa on_message ---

    # --- Add Audio Track ---
    audio_send_task = None
    if SEND_AUDIO_ENABLED:
        try:
            file_audio_track = FileAudioTrack(audio_file)
            pc.addTrack(file_audio_track)
            logger.info("Added file audio track (will wait for session.created).")

            async def audio_sender_wrapper(track):
                 try:
                     await track._stopped.wait()
                     logger.info("Audio track sender finished or was stopped.")
                 except asyncio.CancelledError:
                     logger.info("Audio sender wrapper cancelled.")
                 except Exception as e:
                     logger.error(f"Error in audio sender wrapper: {e}", exc_info=True)

            audio_send_task = asyncio.create_task(audio_sender_wrapper(file_audio_track))
            tasks.add(audio_send_task)
            audio_send_task.add_done_callback(tasks.discard)

        except FileNotFoundError:
            logger.error(f"Cannot start test: Audio file not found at {audio_file}")
            await pc.close()
            return
        except Exception as e:
            logger.error(f"Failed to initialize or add audio track: {e}", exc_info=True)
            await pc.close()
            return
    else:
        logger.info("Sending audio is disabled.")

    # --- Offer/Answer Exchange and Run ---
    try:
        logger.info("Creating SDP offer...")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        logger.info("Set local description (offer).")

        logger.info(f"Sending offer to {server_url} with model={model_name}...")
        request_url = f"{server_url}/v1/realtime"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                request_url,
                params={"model": model_name},
                content=pc.localDescription.sdp,
                headers={"Content-Type": "text/plain"},
                timeout=20.0
            )
            response.raise_for_status()
            answer_sdp = await response.aread()

        logger.info("Received SDP answer from server.")
        answer = RTCSessionDescription(sdp=answer_sdp.decode(), type="answer")
        await pc.setRemoteDescription(answer)
        logger.info("Set remote description (answer). Connection establishing...")

        # --- SỬA LỖI CHỜ ICE ---
        logger.info("Waiting for ICE connection completion...")
        try:
            while pc.iceConnectionState not in ["completed", "connected"]:
                logger.debug(f"Waiting for ICE connection state '{pc.iceConnectionState}'...")
                await asyncio.sleep(0.1)
                if pc.iceConnectionState == "failed":
                    logger.error("ICE Connection failed during wait.")
                    raise ConnectionError("ICE Failed")
                if pc.iceConnectionState == "closed":
                    logger.warning("ICE Connection closed during wait.")
                    raise ConnectionError("ICE Closed")
            logger.info(f"ICE connection state reached: {pc.iceConnectionState}")
        except Exception as ice_e:
            logger.error(f"Error during ICE connection wait: {ice_e}")
            raise ConnectionError(f"ICE Error: {ice_e}")
        # --- KẾT THÚC SỬA LỖI CHỜ ICE ---


        logger.info("Waiting for data channel to open...")
        try:
            # Tăng timeout chờ data channel nếu cần
            async with asyncio.timeout(15): # Chờ tối đa 15 giây
                while channel.readyState != "open":
                    logger.debug(f"Waiting for data channel state '{channel.readyState}'...")
                    await asyncio.sleep(0.1)
                    if pc.connectionState in ["failed", "closed", "disconnected"]:
                        raise ConnectionError("Connection closed while waiting for data channel")
            logger.info("Data channel is open.")
        except TimeoutError: # Bắt lỗi timeout cụ thể
             logger.error("Timeout waiting for data channel!")
             raise ConnectionError("Data Channel Timeout")
        # --- Kết thúc đợi ---


        # --- Wait for audio to finish sending ---
        if audio_send_task:
            logger.info("Waiting for audio file transmission to complete...")
            try:
                await asyncio.wait_for(audio_send_task, timeout=None) # Chờ không giới hạn
                logger.info("Audio file sending task completed.")
            except asyncio.CancelledError:
                 logger.info("Audio sending task was cancelled.")
            except Exception as audio_e:
                 logger.error(f"Error during audio sending wait: {audio_e}")

            # --- GỬI COMMIT EVENT NGAY SAU KHI AUDIO XONG ---
            logger.info("Attempting to send InputAudioBufferCommitEvent...")
            if current_session_id and channel.readyState == "open":
                # QUAN TRỌNG: Vẫn giữ giả định item_id = current_session_id
                # Xem lại comment ở phiên bản MediaPlayer nếu cần điều chỉnh
                commit_event = InputAudioBufferCommitEvent(
                    type="input_audio_buffer.commit",
                    item_id=current_session_id # Giả định item_id = session_id ban đầu
                )

                logger.info(f">>> Sending DC: {commit_event.type} for item_id={commit_event.item_id}")
                try:
                    channel.send(commit_event.model_dump_json())
                    logger.info("Commit event sent.")
                    logger.info("Waiting after commit for server processing (e.g., 10s)...")
                    await asyncio.sleep(10) # Chờ server xử lý
                except Exception as e:
                    logger.error(f"Error sending commit event: {e}")
            elif not current_session_id:
                 logger.warning("Session ID not captured, cannot send commit event.")
            elif channel.readyState != "open":
                 logger.warning(f"Cannot send commit event, data channel state is {channel.readyState}")
            # --- KẾT THÚC GỬI COMMIT ---

        else:
            logger.info("Not sending audio. Running for 10 seconds...")
            await asyncio.sleep(10) # Chạy một lúc nếu không gửi audio


        # Chờ thêm một chút cho các sự kiện cuối cùng (nếu có)
        logger.info("Waiting a final moment for any remaining events...")
        await asyncio.sleep(2)

        logger.info("Finished listening.")
        logger.info("-" * 30)
        logger.info("Full transcript received:")
        logger.info(full_transcript if full_transcript else "[No transcript received]") # In thông báo nếu trống
        logger.info("-" * 30)

    except ConnectionError as e:
         logger.error(f"Setup failed due to connection issue: {e}")
    except httpx.RequestError as e:
        logger.error(f"HTTP request failed: {e}")
    except httpx.HTTPStatusError as e:
         logger.error(f"HTTP error response {e.response.status_code} while sending offer: {e.response.text}")
    except Exception as e:
        logger.exception(f"An error occurred during the main execution loop: {e}")
    finally:
        # --- Cleanup ---
        logger.info("Shutting down...")

        if file_audio_track and not file_audio_track._stopped.is_set():
            logger.info("Explicitly stopping file audio track during cleanup.")
            await file_audio_track.stop() # await để đảm bảo dừng

        active_tasks = [t for t in tasks if not t.done()]
        if active_tasks:
             logger.debug(f"Cancelling {len(active_tasks)} remaining tasks...")
             for task in active_tasks:
                 task.cancel()
             # Chờ các task bị hủy hoàn thành
             await asyncio.gather(*active_tasks, return_exceptions=True)
             logger.debug("Gathered cancelled tasks.")

        if pc and pc.connectionState != "closed":
             logger.info("Closing PeerConnection.")
             await pc.close()
             logger.info("PeerConnection closed.")
        else:
            logger.info("PeerConnection already closed or not initialized.")


# --- Run the Script (không đổi) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC Client Test Tool (FileAudioTrack)") # Sửa description
    parser.add_argument(
        "--server", default="http://localhost:8000", help="Server base URL (e.g., http://localhost:8000)"
    )
    parser.add_argument(
        "--file", default="generated_429000_long.wav", help="Path to the audio file to send (e.g., audio.wav)"
    )
    parser.add_argument(
        # "--model", default="erax-ai/EraX-WoW-Turbo-V1.1-CT2",
        "--model", default="Systran/faster-whisper-large-v3",
        help="Model name parameter for the server"
    )
    parser.add_argument(
        "--no-send-audio", action="store_false", dest="send_audio",
        help="Disable sending audio from the file"
    )
    parser.add_argument(
        "--no-receive-audio", action="store_false", dest="receive_audio",
        help="Disable receiving/consuming audio from the server"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)

    SEND_AUDIO_ENABLED = args.send_audio
    RECEIVE_AUDIO_ENABLED = args.receive_audio

    # Kiểm tra file tồn tại trước khi chạy
    if SEND_AUDIO_ENABLED and not Path(args.file).is_file():
         logger.error(f"Audio file not found at: {args.file}")
    else:
        try:
            asyncio.run(run_test(args.server, args.model, args.file))
        except KeyboardInterrupt:
            logger.info("Test interrupted by user.")
        except FileNotFoundError as e: # Bắt lỗi này nếu Path() kiểm tra sai
            logger.error(f"File Error: {e}")
        except ConnectionError as e: # Bắt lỗi kết nối từ run_test
             logger.error(f"Connection Error during test: {e}")
        except Exception as e:
            logger.exception(f"Unhandled exception in main execution: {e}")

    logger.info("Script finished.")

    #(speaches) (base) aiserver@server-vllm-2:~/speaches_t5tts$ python test_webrtc_client.py --file generated_429000_long.wav --model erax-ai/EraX-WoW-Turbo-V1.1-CT2  --server  http://localhost:8000