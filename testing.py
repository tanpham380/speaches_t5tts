# testing.py
import asyncio
import base64
from fractions import Fraction
import io
import json
import logging
import time
import argparse
import ssl
from pathlib import Path
import random # <<< Vẫn giữ import này cũng được, không dùng nữa thôi

import httpx
import websockets
from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
)
# Bỏ MediaPlayer, MediaRelay nếu không dùng trực tiếp
from pydub import AudioSegment
from pydub.utils import make_chunks
from av import AudioFrame
import numpy as np

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("testing")

# --- Cấu hình Chunk và Pause ---
CHUNK_DURATION_MS = 1000 # <<< Giữ nguyên chunk nhỏ
# MIN_PAUSE_MS = 300    # <<< Không dùng nữa
# MAX_PAUSE_MS = 1000   # <<< Không dùng nữa
WAIT_AFTER_SEND_SECONDS = 20 # <<< Giảm thời gian chờ xuống hợp lý hơn
# --- Hết cấu hình ---

# WebRTC expects 48kHz stereo s16 input track based on server's audio_receiver
RTC_TARGET_SAMPLE_RATE = 48000
RTC_TARGET_CHANNELS = 2
RTC_TARGET_SAMPLE_WIDTH = 2 # s16 = 2 bytes

# WebSocket sends raw audio bytes. Based on server logic (RTC receiver resamples to 24k),
# sending 24k mono s16 seems appropriate for the input_audio_buffer.append event.
WS_TARGET_SAMPLE_RATE = 24000
WS_TARGET_CHANNELS = 1
WS_TARGET_SAMPLE_WIDTH = 2 # s16 = 2 bytes

# Global variable to signal audio sending completion for RTC
rtc_audio_sending_done = asyncio.Event()
# --- Thêm Event để quản lý đóng kết nối RTC ---
rtc_connection_closed_event = asyncio.Event()


# --- Helper: Audio Processing (Xoá phần giả lập pause) ---
async def stream_audio_chunks(
    file_path: str,
    chunk_duration_ms: int,
    target_sample_rate: int,
    target_channels: int,
    target_sample_width: int, # Bytes per sample
    target_format: str = 's16le' # 's16le' for signed 16-bit little-endian
):
    """
    Asynchronously yields audio chunks from a file, resampled and timed,
    *without* adding artificial pauses.
    """
    logger.info(f"Loading audio file: {file_path}")
    try:
        audio = AudioSegment.from_wav(file_path)
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        return

    logger.info(
        f"Original audio: {audio.frame_rate}Hz, {audio.channels}ch, "
        f"{audio.sample_width * 8}-bit"
    )

    # Resample and set channels if necessary
    if (
        audio.frame_rate != target_sample_rate
        or audio.channels != target_channels
        # sample_width check
    ):
        logger.info(
            f"Resampling audio to {target_sample_rate}Hz, {target_channels}ch, "
            f"{target_sample_width * 8}-bit"
        )
        try:
            audio = audio.set_frame_rate(target_sample_rate)
            audio = audio.set_channels(target_channels)
            if audio.sample_width != target_sample_width:
                 logger.warning(f"Original sample width {audio.sample_width*8}-bit differs from target {target_sample_width*8}-bit. Ensuring export matches.")

        except Exception as e:
            logger.error(f"Error during audio resampling: {e}")
            return

    logger.info(f"Audio prepared for streaming. Duration: {audio.duration_seconds:.2f}s")
    chunks = make_chunks(audio, chunk_duration_ms)
    start_time = time.perf_counter()
    total_chunks = len(chunks)
    # min_pause_sec = MIN_PAUSE_MS / 1000.0 # <<< Không dùng nữa
    # max_pause_sec = MAX_PAUSE_MS / 1000.0 # <<< Không dùng nữa

# Trong hàm stream_audio_chunks của testing.py

    for i, chunk in enumerate(chunks):
        # --- LƯU DEBUG TRỰC TIẾP TỪ CHUNK (Giữ nguyên) ---
        # try:
        #     pydub_direct_save_dir = Path("./client_debug_audio/pydub_direct_export")
        #     pydub_direct_save_dir.mkdir(parents=True, exist_ok=True)
        #     # Đặt tên file rõ ràng hơn một chút (ví dụ: dùng rate/bit từ chunk nếu cần)
        #     pydub_direct_save_path = pydub_direct_save_dir / f"pydub_chunk_{i:04d}_{chunk.frame_rate//1000}k_{chunk.sample_width*8}bit.wav"
        #     chunk.export(str(pydub_direct_save_path), format="wav")
        # except Exception as pydub_export_err:
        #     logger.error(f"Failed to export direct pydub chunk {i+1}: {pydub_export_err}")


        raw_data = chunk.raw_data
        logger.debug(f"Chunk {i+1} raw_data length: {len(raw_data)}")

        # --- Kiểm tra độ dài raw_data (Giữ nguyên hoặc cải thiện nếu cần) ---
        expected_raw_data_len = int((len(chunk) / 1000.0) * chunk.frame_rate * chunk.frame_width)
        if abs(len(raw_data) - expected_raw_data_len) > chunk.frame_width * 2: # Sai số nhỏ cho phép
            logger.warning(f"Chunk {i+1}: Potential mismatch between pydub length and raw_data length. "
                           f"Pydub ms={len(chunk)}, Expected bytes ~{expected_raw_data_len}, Got raw_data bytes={len(raw_data)}")

        if not raw_data: continue

        # --- Phần lưu debug từ raw_data (SỬA Ở ĐÂY) ---
        try:
            # ***** SỬ DỤNG THUỘC TÍNH THỰC TẾ CỦA CHUNK *****
            chunk_segment = AudioSegment(
                data=raw_data,
                sample_width=chunk.sample_width,  # <<< Lấy từ chunk
                frame_rate=chunk.frame_rate,    # <<< Lấy từ chunk
                channels=chunk.channels          # <<< Lấy từ chunk
            )
            # # Cập nhật tên file để phản ánh đúng định dạng đã lưu
            # client_raw_save_path = Path(f"./client_debug_audio/from_raw_data/client_chunk_{i:04d}_{chunk.frame_rate//1000}k_{chunk.sample_width*8}bit_{chunk.channels}ch.wav")
            # client_raw_save_path.parent.mkdir(parents=True, exist_ok=True)
            # chunk_segment.export(str(client_raw_save_path), format="wav")

        except Exception as client_save_err:
            # Log lỗi rõ ràng hơn nếu vẫn xảy ra
            logger.error(f"Failed to save client chunk from raw data {i+1} using chunk properties "
                         f"(w={chunk.sample_width}, r={chunk.frame_rate}, c={chunk.channels}): {client_save_err}")

        actual_width = chunk.sample_width
        actual_rate = chunk.frame_rate
        actual_channels = chunk.channels

        data_to_yield = raw_data

        if (actual_width != target_sample_width or
            actual_rate != target_sample_rate or
            actual_channels != target_channels):

            # logger.warning(f"Chunk {i+1} raw_data format ({actual_width*8}-bit, {actual_rate}Hz, {actual_channels}ch) "
            #                f"differs from target ({target_sample_width*8}-bit, {target_sample_rate}Hz, {target_channels}ch). "
            #                f"Attempting final conversion before yielding.")
            try:
                temp_segment = AudioSegment(data=raw_data, sample_width=actual_width, frame_rate=actual_rate, channels=actual_channels)

                if temp_segment.frame_rate != target_sample_rate:
                    temp_segment = temp_segment.set_frame_rate(target_sample_rate)
                if temp_segment.channels != target_channels:
                    temp_segment = temp_segment.set_channels(target_channels)
                buffer = io.BytesIO()
                temp_segment.export(buffer, format="s16le")
                data_to_yield = buffer.getvalue()
                # logger.debug(f"Chunk {i+1} successfully converted to target format before yielding. New raw_data length: {len(data_to_yield)}")

            except Exception as conversion_err:
                logger.error(f"Chunk {i+1}: Failed final conversion to target format before yielding: {conversion_err}. Yielding unconverted data.")
                data_to_yield = raw_data
                # yield raw_data # Yield dữ liệu gốc nếu không convert được

        # Kiểm tra lại số byte của data_to_yield trước khi yield
        bytes_per_sample_final = target_sample_width # Kể cả sau khi convert
        channels_final = target_channels
        frame_size_final = channels_final * bytes_per_sample_final
        if len(data_to_yield) % frame_size_final != 0:
             num_frames = len(data_to_yield) // frame_size_final
             logger.warning(f"Chunk {i+1} final data size {len(data_to_yield)} for yielding is not perfectly divisible by final frame size ({frame_size_final}). Truncating.")
             data_to_yield = data_to_yield[:num_frames * frame_size_final] # Cắt bỏ phần thừa

        if len(data_to_yield) == 0:
            logger.debug(f"Skipping empty chunk {i+1} after final processing.")
            continue

        # Yield dữ liệu đã (hy vọng) đúng định dạng target
        yield data_to_yield

        # --- Timing delay (Giữ nguyên) ---
        current_cumulative_duration_sec = (i + 1) * (chunk_duration_ms / 1000.0)
        expected_playback_time = start_time + current_cumulative_duration_sec
        current_time = time.perf_counter()
        playback_delay = expected_playback_time - current_time
        if playback_delay > 0:
            await asyncio.sleep(playback_delay)

    logger.info("Finished yielding all audio chunks.")


class DummyAudioTrack(MediaStreamTrack):
    """
    A MediaStreamTrack that reads audio from our async generator (without extra pauses)
    and yields AV AudioFrames.
    """
    kind = "audio"

    def __init__(self, audio_generator):
        super().__init__()
        self._generator = audio_generator # Generator không còn pause nhân tạo
        self._queue = asyncio.Queue()
        self._producer_task = None
        self._timestamp = 0 # Initialize timestamp here
        logger.info("DummyAudioTrack initialized.")

    async def _run_producer(self):
        logger.info("Audio frame producer task started.")
        async for chunk_bytes in self._generator: # Generator handles timing (chỉ playback)
            if not chunk_bytes:
                 continue
            samples_np = np.frombuffer(chunk_bytes, dtype=np.int16)
            num_samples_per_channel = len(samples_np) // RTC_TARGET_CHANNELS
            if num_samples_per_channel == 0:
                 logger.warning("Received chunk resulted in 0 samples per channel, skipping frame.")
                 continue
            try:
                frame = AudioFrame(
                    format="s16",
                    layout="stereo" if RTC_TARGET_CHANNELS == 2 else "mono",
                    samples=num_samples_per_channel
                )
                frame.sample_rate = RTC_TARGET_SAMPLE_RATE
                frame.planes[0].update(samples_np.tobytes())
                frame.pts = self._timestamp
                frame.time_base = Fraction(1, RTC_TARGET_SAMPLE_RATE)
                self._timestamp += num_samples_per_channel
                await self._queue.put(frame)
            except Exception as frame_error:
                 logger.error(f"Error creating or queuing AudioFrame: {frame_error}")

        await self._queue.put(None)
        logger.info("Audio frame producer task finished.")
        rtc_audio_sending_done.set()


    async def recv(self):
        if self._producer_task is None:
            self._producer_task = asyncio.create_task(self._run_producer())
            logger.info("Started audio frame producer task.")

        frame = await self._queue.get()
        if frame is None:
            logger.info("Received EOS signal in DummyAudioTrack.recv.")
            self.stop()
            raise asyncio.CancelledError("Audio stream ended")
        return frame

    def stop(self):
        super().stop()
        if self._producer_task and not self._producer_task.done():
            self._producer_task.cancel()
            logger.info("Cancelled audio producer task.")


async def test_websocket(server_url: str, file_path: str, model: str, api_key: str | None):
    """Tests the WebSocket endpoint without simulated pauses and collects the final transcript."""
    ws_url = f"{server_url.replace('http', 'ws')}/v1/realtime?model={model}"
    logger.info(f"Connecting to WebSocket: {ws_url}")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    final_transcript = "[Transcript retrieval failed or task did not complete properly]"
    receive_task = None

    try:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        async with websockets.connect(
            ws_url,
            # extra_headers=headers,
            # ssl=ssl_context
        ) as websocket:
            logger.info("WebSocket connected.")
            receive_task = asyncio.create_task(receive_ws_messages(websocket))
            send_task = asyncio.create_task(
                send_ws_audio(
                    websocket,
                    file_path,
                    CHUNK_DURATION_MS,
                    WS_TARGET_SAMPLE_RATE,
                    WS_TARGET_CHANNELS,
                    WS_TARGET_SAMPLE_WIDTH
                )
            )
            await send_task
            logger.info(f"Finished sending audio. Waiting {WAIT_AFTER_SEND_SECONDS}s for final server responses...")
            try:
                final_transcript = await asyncio.wait_for(receive_task, timeout=WAIT_AFTER_SEND_SECONDS)
                logger.info("Receive task completed and returned combined transcript.")
            except asyncio.TimeoutError:
                logger.warning(f"Stopped waiting for responses after {WAIT_AFTER_SEND_SECONDS}s timeout. Cancelling receive task.")
                if receive_task and not receive_task.done():
                    receive_task.cancel()
                    try:
                        final_transcript = await receive_task
                        logger.info("Received partial transcript after cancellation.")
                    except asyncio.CancelledError:
                        logger.info("Receive task confirmed cancelled after timeout.")
            except asyncio.CancelledError:
                logger.warning("Receive task was cancelled externally.")
                try:
                    final_transcript = await receive_task
                except asyncio.CancelledError:
                     pass
            except Exception as e:
                 logger.error(f"Receive task failed with exception: {e}")
                 if receive_task and receive_task.done() and not receive_task.cancelled():
                     try:
                        if receive_task.exception():
                             logger.error(f"Receive task completed with error: {receive_task.exception()}")
                        else:
                             final_transcript = receive_task.result()
                             logger.info("Received partial transcript after receive task error.")
                     except Exception as get_result_err:
                          logger.error(f"Could not get result after receive task error: {get_result_err}")

            logger.info("WebSocket test scenario finished.")

    except websockets.exceptions.ConnectionClosedOK:
        logger.info("WebSocket connection closed normally by server.")
    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"WebSocket connection failed: Status {e.status_code}")
    except ConnectionRefusedError:
        logger.error(f"Connection refused. Is the server running at {server_url}?")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.info(f"\n--- FINAL COMBINED WS TRANSCRIPT ---\n{final_transcript}\n------------------------------------")
        if receive_task and not receive_task.done():
            logger.info("Attempting final cancellation of receive task.")
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass



async def receive_ws_messages(websocket) -> str:
    """
    Receives messages, logs them, and collects completed transcripts
    to return a single combined string at the end.
    """
    received_transcripts = []
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                event_type = data.get("type", "unknown")
                logger.info(f"<<< WS RECV ({event_type}): {json.dumps(data, indent=2)}")
                if event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = data.get("transcript")
                    if transcript:
                        logger.info(f"---> WS Transcript Received: {transcript}")
                        received_transcripts.append(transcript.strip())
            except json.JSONDecodeError:
                logger.warning(f"<<< WS RECV non-JSON: {message}")
            except Exception as e:
                logger.error(f"Error processing received WS message: {e}")

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"WebSocket connection closed by server while receiving (code: {e.code}, reason: {e.reason}). Loop terminated.")
    except asyncio.CancelledError:
        logger.info("WS receive task cancelled.")
        logger.warning("Returning potentially incomplete transcript due to cancellation.")
    except Exception as e:
        logger.error(f"Unhandled error in WS receive task main loop: {e}", exc_info=True)
        logger.error("Returning potentially incomplete transcript due to unhandled error.")

    final_text = ' '.join(received_transcripts)
    logger.info(f"Receive loop finished/exited. Final combined transcript length: {len(final_text)}")
    return final_text



async def send_ws_audio(
    websocket,
    file_path: str,
    chunk_duration_ms: int,
    target_sample_rate: int,
    target_channels: int,
    target_sample_width: int
):
    """Streams audio chunks to the WebSocket server using the generator."""
    logger.info("Starting to send audio chunks...")
    sent_chunk_count = 0
    try:
        audio_generator = stream_audio_chunks( # Gọi generator đã cập nhật (không pause)
            file_path,
            chunk_duration_ms,
            target_sample_rate,
            target_channels,
            target_sample_width
        )
        async for chunk_bytes in audio_generator:
            b64_audio = base64.b64encode(chunk_bytes).decode("utf-8")
            event = { "type": "input_audio_buffer.append", "audio": b64_audio }
            message = json.dumps(event)
            await websocket.send(message)
            sent_chunk_count += 1
            logger.debug(f"Sent chunk {sent_chunk_count}")

        logger.info(f"Finished sending all {sent_chunk_count} audio chunks via WebSocket.")
    except websockets.exceptions.ConnectionClosed as e:
        logger.warning(f"WebSocket connection closed by server while sending (code: {e.code}, reason: {e.reason}).")
    except asyncio.CancelledError:
        logger.info("WS send task cancelled.")
        raise
    except Exception as e:
        logger.error(f"Error in WS send task: {e}", exc_info=True)
        raise


async def test_webrtc(server_url: str, file_path: str, model: str, api_key: str | None):
    """Tests the WebRTC endpoint without simulated pauses."""
    rtc_url = f"{server_url}/v1/realtime?model={model}"
    logger.info(f"Initiating WebRTC connection to: {rtc_url}")

    rtc_audio_sending_done.clear()
    rtc_connection_closed_event.clear()

    pc = RTCPeerConnection()
    data_channel = pc.createDataChannel("events")
    logger.info("RTC PeerConnection and DataChannel created.")

    received_rtc_transcripts = [] # <<< Thêm list để thu thập transcript RTC

    @data_channel.on("open")
    def on_open():
        logger.info("RTC DataChannel 'events' opened.")

    @data_channel.on("message")
    def on_message(message):
        try:
            data = json.loads(message)
            event_type = data.get("type", "unknown")
            logger.info(f"<<< RTC RECV ({event_type}): {json.dumps(data, indent=2)}")
            if event_type == "conversation.item.input_audio_transcription.completed":
               transcript = data.get("transcript")
               if transcript:
                   logger.info(f"---> RTC Transcript Received: {transcript}")
                   received_rtc_transcripts.append(transcript.strip()) # <<< Thu thập transcript
        except json.JSONDecodeError:
            logger.warning(f"<<< RTC RECV non-JSON: {message}")
        except Exception as e:
            logger.error(f"Error processing received RTC message: {e}")

    @data_channel.on("close")
    def on_close():
         logger.info("RTC DataChannel 'events' closed.")
         rtc_connection_closed_event.set()

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"RTC ICE connection state is {pc.iceConnectionState}")
        if pc.iceConnectionState in ("failed", "closed", "disconnected"):
            rtc_connection_closed_event.set()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
         logger.info(f"RTC Peer connection state is {pc.connectionState}")
         if pc.connectionState in ("failed", "closed", "disconnected"):
             rtc_connection_closed_event.set()

    audio_generator = stream_audio_chunks( # Generator không pause
        file_path,
        CHUNK_DURATION_MS,
        RTC_TARGET_SAMPLE_RATE,
        RTC_TARGET_CHANNELS,
        RTC_TARGET_SAMPLE_WIDTH
    )
    audio_track = DummyAudioTrack(audio_generator)
    pc.addTrack(audio_track)
    logger.info("Audio track added to PeerConnection.")

    final_rtc_transcript = "[RTC Transcript retrieval incomplete]" # <<< Biến cho kết quả RTC

    try:
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        logger.info("RTC Offer created and local description set.")

        headers = {"Content-Type": "application/sdp"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(verify=False) as client:
            logger.info(f"Sending offer to {rtc_url}...")
            response = await client.post(
                rtc_url,
                content=pc.localDescription.sdp.encode("utf-8"),
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            answer_sdp = response.text
            logger.info("Received Answer SDP from server.")

        answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
        await pc.setRemoteDescription(answer)
        logger.info("Remote description (answer) set.")

        logger.info("Waiting for audio streaming to finish...")
        await rtc_audio_sending_done.wait()
        logger.info(f"Audio sending finished. Waiting up to {WAIT_AFTER_SEND_SECONDS}s for final server responses or connection close...")

        try:
            await asyncio.wait_for(
                rtc_connection_closed_event.wait(),
                timeout=WAIT_AFTER_SEND_SECONDS
            )
            if rtc_connection_closed_event.is_set():
                logger.info("Connection closed event received.")
        except asyncio.TimeoutError:
            logger.info(f"Stopped waiting for RTC responses/closure after {WAIT_AFTER_SEND_SECONDS}s timeout.")
        except Exception as wait_err:
             logger.error(f"Error while waiting for RTC closure/timeout: {wait_err}")

        # <<< Ghép transcript RTC sau khi chờ
        final_rtc_transcript = ' '.join(received_rtc_transcripts)

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error sending Offer: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"HTTP Request Error: {e}")
    except ConnectionRefusedError:
         logger.error(f"Connection refused for signaling. Is the server running at {server_url}?")
    except Exception as e:
        logger.error(f"WebRTC Error: {e}", exc_info=True)
    finally:
        # <<< Log kết quả RTC
        logger.info(f"\n--- FINAL COMBINED RTC TRANSCRIPT ---\n{final_rtc_transcript}\n------------------------------------")
        logger.info("Closing RTC PeerConnection.")
        if pc.connectionState != "closed":
             await pc.close()
        rtc_audio_sending_done.clear()
        rtc_connection_closed_event.clear()
        logger.info("WebRTC test scenario finished.")


# --- Main Execution (Giữ nguyên) ---
async def main():
    parser = argparse.ArgumentParser(description="Test Realtime Endpoints")
    parser.add_argument(
        "protocol", choices=["ws", "rtc"], help="Protocol to test (ws or rtc)"
    )
    parser.add_argument(
        "--server", default="http://127.0.0.1:8000", help="Server base URL"
    )
    parser.add_argument(
        "--file", default="generated_429000_long.wav", help="Path to the WAV audio file"
    )
    parser.add_argument(
        "--model", default="Systran/faster-whisper-large-v3",
        help="Model name to use"
    )
    parser.add_argument("--api-key", default=None, help="API key if required")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logging.getLogger("websockets").setLevel(logging.INFO)
        logging.getLogger("aiortc").setLevel(logging.INFO)

    audio_file_path = Path(args.file)
    if not audio_file_path.is_file():
        logger.error(f"Audio file not found: {args.file}")
        return

    if args.protocol == "ws":
        await test_websocket(args.server, str(audio_file_path), args.model, args.api_key)
    elif args.protocol == "rtc":
        await test_webrtc(args.server, str(audio_file_path), args.model, args.api_key)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    


    # # --- End of the script ---
    # # Note: The script will exit after the main function completes.
    # model = "erax-ai/EraX-WoW-Turbo-V1.1-CT2"

    #python testing.py rtc --file generated_429000_long.wav --model Systran/faster-whisper-large-v3 --server http://localhost:8000
    #python testing.py ws --file generated_429000_long.wav --model erax-ai/EraX-WoW-Turbo-V1.1-CT2 --server http://localhost:8000
