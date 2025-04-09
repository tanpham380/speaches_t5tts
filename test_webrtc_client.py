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
)
from aiortc.contrib.media import MediaBlackhole
from av import AudioFrame
import httpx
from pydub import AudioSegment
from pydub.utils import make_chunks

# --- Configuration ---
SEND_AUDIO_ENABLED = True
RECEIVE_AUDIO_ENABLED = True
RTC_SEND_SAMPLE_RATE = 48000
RTC_SEND_CHANNELS = 2
RTC_SEND_FORMAT_AV = "s16"
RTC_SEND_LAYOUT_AV = "stereo"
RTC_SEND_SAMPLE_WIDTH = 2
RTC_SEND_DTYPE = np.int16
AUDIO_CHUNK_DURATION_MS = 1000 # Giữ nguyên chunk duration
AUDIO_PTIME = AUDIO_CHUNK_DURATION_MS / 1000.0
SAMPLES_PER_FRAME = int(RTC_SEND_SAMPLE_RATE * AUDIO_PTIME)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebRTCClientTest")
logging.getLogger("pydub").setLevel(logging.WARNING)
# logging.getLogger("aiortc").setLevel(logging.DEBUG) # Bật lại nếu cần debug sâu
# logging.getLogger("aioice").setLevel(logging.DEBUG)

# --- Global Event to signal session created ---
session_created_event = asyncio.Event() # <--- EVENT MỚI

# --- Helper: Audio Processing (stream_audio_chunks - không đổi) ---
async def stream_audio_chunks(
    file_path: str,
    chunk_duration_ms: int,
    target_sample_rate: int,
    target_channels: int,
    target_sample_width: int, # Bytes per sample
    target_format: str = 's16le' # 's16le' for signed 16-bit little-endian raw bytes
):
    # ... (Giữ nguyên code của hàm này) ...
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
        logger.info(f"Audio prepared for streaming: {audio.frame_rate}Hz, {audio.channels}ch, {audio.sample_width*8}-bit")

    chunks = make_chunks(audio, chunk_duration_ms)
    start_time = time.perf_counter()
    total_chunks = len(chunks)
    playback_time_offset = 0.0

    logger.info(f"Ready to stream {total_chunks} chunks of {chunk_duration_ms}ms each (waiting for session)...") # Sửa log

    # --- CHỜ SESSION ĐƯỢC TẠO TRƯỚC KHI BẮT ĐẦU STREAM ---
    logger.info("stream_audio_chunks: Waiting for session_created_event...")
    await session_created_event.wait() # <--- ĐỢI EVENT
    logger.info("stream_audio_chunks: session_created_event received! Starting audio stream.")
    start_time = time.perf_counter() # Reset start time after wait
    # -----------------------------------------------------

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


# --- File Audio Track to Send (recv - không cần thay đổi nhiều) ---
class FileAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Khởi tạo generator ngay lập tức, nó sẽ tự đợi event bên trong
        self._generator = stream_audio_chunks(
            file_path=str(self.file_path),
            chunk_duration_ms=AUDIO_CHUNK_DURATION_MS,
            target_sample_rate=RTC_SEND_SAMPLE_RATE,
            target_channels=RTC_SEND_CHANNELS,
            target_sample_width=RTC_SEND_SAMPLE_WIDTH,
            target_format='s16le'
        )
        self._frame_pts = 0
        self._first_frame = True # Cờ để kiểm tra lần gọi đầu
        self._start_time = None # Sẽ được đặt khi frame đầu tiên được gửi
        self._stopped = asyncio.Event()

    async def recv(self) -> AudioFrame:
        if self._stopped.is_set():
             raise StopAsyncIteration

        # Không cần đợi event ở đây nữa vì generator sẽ tự đợi

        try:
            # Get the next chunk of raw bytes from the generator
            chunk_bytes = await self._generator.__anext__()
            if self._first_frame:
                logger.info("FileAudioTrack: Received first audio chunk from generator.")
                self._start_time = time.time() # Đặt start_time khi frame đầu tiên thực sự được gửi
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
            samples_reshaped = np.ascontiguousarray(samples_np.reshape((samples_in_chunk, RTC_SEND_CHANNELS)))

            frame = AudioFrame(format=RTC_SEND_FORMAT_AV, layout=RTC_SEND_LAYOUT_AV, samples=samples_in_chunk)
            frame.sample_rate = RTC_SEND_SAMPLE_RATE
            frame.pts = self._frame_pts

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
             await self._generator.aclose()


# --- Main Test Function (run_test) ---
async def run_test(server_url: str, model_name: str, audio_file: str):
    logger.info("Starting WebRTC client test...")
    # ... (logging server, model, file) ...
    session_created_event.clear() # Đảm bảo event được reset khi bắt đầu test mới

    config = RTCConfiguration()
    pc = RTCPeerConnection(configuration=config)
    tasks = set()
    file_audio_track = None

    # --- Event Handlers (on_iceconnectionstatechange, on_connectionstatechange, on_track - không đổi) ---
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state is {pc.iceConnectionState}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed", "disconnected"]:
             logger.warning(f"Peer connection state changed to {pc.connectionState}. Stopping audio send if active.")
             if file_audio_track and not file_audio_track._stopped.is_set():
                 # Gọi stop không đồng bộ để tránh block handler
                 asyncio.create_task(file_audio_track.stop())


    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"Data channel '{channel.label}' created by remote (unexpected)")

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
                if recorder and not task.done():
                     # Không cần await recorder.stop() ở đây nữa vì task wrapper sẽ xử lý
                     logger.info("Recorder will be stopped by task cancellation.")
        elif track.kind == "video":
             logger.warning("Received unexpected video track")

    # --- Create Data Channel ---
    channel = pc.createDataChannel("client_data_channel")
    logger.info(f"Created data channel '{channel.label}'")

    @channel.on("open")
    def on_open():
        logger.info(f"Data channel '{channel.label}' is open")

    @channel.on("close")
    def on_close():
         logger.info(f"Data channel '{channel.label}' is closed")

    @channel.on("message")
    def on_message(message):
        try:
            event = json.loads(message)
            event_type = event.get("type", "unknown")
            logger.info(f"< Server DC ({event_type}): {json.dumps(event)}")
            # --- KÍCH HOẠT EVENT KHI NHẬN session.created ---
            if event_type == "session.created":
                logger.info(">>> Received session.created! Setting event to start audio streaming. <<<")
                session_created_event.set() # <--- KÍCH HOẠT EVENT
            # --------------------------------------------------
        except json.JSONDecodeError:
            logger.warning(f"< Server DC non-JSON: {message}")

    # --- Add Audio Track ---
    audio_send_task = None
    if SEND_AUDIO_ENABLED:
        try:
            # Khởi tạo track, nó sẽ tự đợi event bên trong generator
            file_audio_track = FileAudioTrack(audio_file)
            pc.addTrack(file_audio_track)
            logger.info("Added file audio track (will wait for session.created).")

            # Wrapper vẫn cần thiết để biết khi nào audio thực sự gửi xong
            async def audio_sender_wrapper(track):
                 try:
                     # Đợi event _stopped của track được set (khi generator kết thúc)
                     await track._stopped.wait()
                     logger.info("Audio track sender finished or was stopped.")
                 except asyncio.CancelledError:
                     logger.info("Audio sender wrapper cancelled.")
                 except Exception as e:
                     logger.error(f"Error in audio sender wrapper: {e}", exc_info=True)
                 finally:
                     # Không cần gọi track.stop() ở đây nữa vì nó tự stop khi generator hết
                     pass

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
        # ... (createOffer, setLocalDescription) ...
        logger.info("Creating SDP offer...")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        logger.info("Set local description (offer).")


        # ... (send offer, receive answer) ...
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
        logger.info("Waiting for connection and session.created event...") # Log chờ đợi

        # --- Wait for audio to finish sending OR connection fail ---
        if audio_send_task:
            # Không cần đợi session_created_event ở đây nữa,
            # audio_sender_wrapper sẽ tự kết thúc khi track._stopped được set
            logger.info("Connection established (or establishing). Audio will start after session.created.")
            logger.info("Waiting for audio file transmission to complete...")
            try:
                done, pending = await asyncio.wait(
                    [audio_send_task], # Chỉ cần đợi task wrapper hoàn thành
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=None
                )
                if audio_send_task in pending:
                     logger.warning("Audio sending wait interrupted.")
                elif audio_send_task.done() and audio_send_task.exception():
                    # Lỗi xảy ra trong audio_sender_wrapper (có thể do generator lỗi)
                    logger.error(f"Audio sending task failed: {audio_send_task.exception()}")
                else:
                    # Hoàn thành bình thường
                    logger.info("Audio file sending task completed successfully.")

            except asyncio.CancelledError:
                 logger.info("Main wait task cancelled.")

            logger.info("Audio sending finished. Waiting 2s for final messages...")
            await asyncio.sleep(2)

        else:
            logger.info("Not sending audio. Running for 10 seconds to test connection...")
            # Nếu không gửi audio, vẫn cần đợi session.created nếu muốn test logic đó
            try:
                await asyncio.wait_for(session_created_event.wait(), timeout=10.0)
                logger.info("Received session.created (audio sending disabled).")
                await asyncio.sleep(5) # Chờ thêm chút
            except asyncio.TimeoutError:
                logger.warning("Did not receive session.created within 10s (audio sending disabled).")

    # ... (except blocks - không đổi) ...
    except httpx.RequestError as e:
        logger.error(f"HTTP request failed: {e}")
    except httpx.HTTPStatusError as e:
         logger.error(f"HTTP error response {e.response.status_code} while sending offer: {e.response.text}")
    except Exception as e:
        logger.exception(f"An error occurred during the test: {e}")
    finally:
        # --- Cleanup (không đổi) ---
        logger.info("Shutting down...")

        if file_audio_track and not file_audio_track._stopped.is_set():
            logger.info("Explicitly stopping file audio track during cleanup.")
            # Gọi không đồng bộ
            asyncio.create_task(file_audio_track.stop())

        for task in list(tasks):
            if task and not task.done():
                logger.debug(f"Cancelling task {task.get_name()}...")
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug("Gathered cancelled tasks.")

        if pc and pc.connectionState != "closed":
             logger.info("Closing PeerConnection.")
             await pc.close()
             logger.info("PeerConnection closed.")
        else:
            logger.info("PeerConnection already closed or not initialized.")


# --- Run the Script (không đổi) ---
if __name__ == "__main__":
    # ... (argparse setup - không đổi) ...
    parser = argparse.ArgumentParser(description="WebRTC Client Test Tool")
    parser.add_argument(
        "--server", default="http://localhost:8000", help="Server base URL (e.g., http://localhost:8000)"
    )
    parser.add_argument(
        "--file", default="generated_429000_long.wav", help="Path to the audio file to send (e.g., audio.wav)"
    )
    parser.add_argument(
        "--model", default="erax-ai/EraX-WoW-Turbo-V1.1-CT2", # Sensible default
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

    SEND_AUDIO_ENABLED = args.send_audio
    RECEIVE_AUDIO_ENABLED = args.receive_audio

    try:
        asyncio.run(run_test(args.server, args.model, args.file))
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except FileNotFoundError as e:
         logger.error(f"File Error: {e}")
    except Exception as e:
         logger.exception(f"Unhandled exception in main execution: {e}")

    logger.info("Script finished.")



    # python test_webrtc_client.py --file generated_429000_long.wav --model erax-ai/EraX-WoW-Turbo-V1.1-CT2  --server  http://localhost:8000
