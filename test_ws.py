# testing.py
import asyncio
import base64
from fractions import Fraction
import io # <<< Cần import io
import json
import logging
import time
import argparse
import ssl
from pathlib import Path
# import random # Không cần nữa

import websockets

from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np # <<< Cần import numpy

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("testing")

# --- Cấu hình Chunk và Pause ---
CHUNK_DURATION_MS = 1000 # <<< Tăng lại chunk duration như đề xuất trước
WAIT_AFTER_SEND_SECONDS = 5
# --- Hết cấu hình ---


# WebSocket sends raw audio bytes. Based on server logic (RTC receiver resamples to 24k),
# sending 24k mono s16 seems appropriate for the input_audio_buffer.append event.
WS_TARGET_SAMPLE_RATE = 24000
WS_TARGET_CHANNELS = 1
WS_TARGET_SAMPLE_WIDTH = 2 # s16 = 2 bytes

# Global variable to signal audio sending completion for RTC
rtc_audio_sending_done = asyncio.Event()
# --- Thêm Event để quản lý đóng kết nối RTC ---
rtc_connection_closed_event = asyncio.Event()


# --- Helper: Audio Processing ---
async def stream_audio_chunks(
    file_path: str,
    chunk_duration_ms: int,
    target_sample_rate: int,
    target_channels: int,
    target_sample_width: int, # Bytes per sample
    target_format: str = 's16le' # 's16le' for signed 16-bit little-endian
):
    """
    Asynchronously yields audio chunks from a file, resampled and timed.
    Prepends 1 second of silence to the very first chunk.
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
    ):
        logger.info(
            f"Resampling audio to {target_sample_rate}Hz, {target_channels}ch, "
            f"{target_sample_width * 8}-bit"
        )
        try:
            audio = audio.set_frame_rate(target_sample_rate)
            audio = audio.set_channels(target_channels)
            # Cảnh báo nếu pydub cần thay đổi sample width (ví dụ từ 32bit xuống 16bit)
            if audio.sample_width != target_sample_width:
                 logger.warning(f"Original sample width {audio.sample_width*8}-bit differs from target {target_sample_width*8}-bit. Pydub will handle export.")

        except Exception as e:
            logger.error(f"Error during audio resampling: {e}")
            return

    logger.info(f"Audio prepared for streaming. Duration: {audio.duration_seconds:.2f}s")
    chunks = make_chunks(audio, chunk_duration_ms)
    start_time = time.perf_counter()
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        # Lấy dữ liệu thô từ chunk (có thể chưa đúng target format)
        raw_data = chunk.raw_data
        if not raw_data: continue

        # Lấy định dạng thực tế của raw_data từ chunk object
        actual_width = chunk.sample_width
        actual_rate = chunk.frame_rate
        actual_channels = chunk.channels

        data_to_yield = raw_data

        # Chuyển đổi sang định dạng target NẾU CẦN THIẾT trước khi yield
        if (actual_width != target_sample_width or
            actual_rate != target_sample_rate or
            actual_channels != target_channels):

            logger.debug(f"Chunk {i+1} format ({actual_width*8}-bit, {actual_rate}Hz, {actual_channels}ch) "
                           f"differs from target ({target_sample_width*8}-bit, {target_sample_rate}Hz, {target_channels}ch). "
                           f"Converting using export(format='{target_format}').")
            try:
                temp_segment = AudioSegment(data=raw_data, sample_width=actual_width, frame_rate=actual_rate, channels=actual_channels)

                # Đảm bảo rate và channels đúng trước khi export format raw
                if temp_segment.frame_rate != target_sample_rate:
                    temp_segment = temp_segment.set_frame_rate(target_sample_rate)
                if temp_segment.channels != target_channels:
                    temp_segment = temp_segment.set_channels(target_channels)

                # Export vào buffer để lấy đúng bytes target
                buffer = io.BytesIO()
                temp_segment.export(buffer, format=target_format) # Sử dụng target_format (ví dụ 's16le')
                data_to_yield = buffer.getvalue()
                logger.debug(f"Chunk {i+1} successfully converted. New data length: {len(data_to_yield)}")

            except Exception as conversion_err:
                logger.error(f"Chunk {i+1}: Failed conversion using export(format='{target_format}'): {conversion_err}. Yielding unconverted data.", exc_info=True)
                data_to_yield = raw_data # Fallback

        # --- THÊM 1 GIÂY IM LẶNG VÀO CHUNK ĐẦU TIÊN ---
        if i == 0:
            logger.info("Prepending 1 second of silence to the first chunk.")
            silence_duration_sec = 1.0
            # Tính số sample cần cho khoảng lặng ở target rate/channels
            num_silence_samples = int(silence_duration_sec * target_sample_rate * target_channels)
            # Tạo mảng numpy chứa các giá trị 0 (im lặng) với kiểu int16
            silence_samples_np = np.zeros(num_silence_samples, dtype=np.int16)
            # Chuyển mảng numpy thành bytes
            silence_bytes = silence_samples_np.tobytes()

            # Ghép bytes im lặng vào đầu dữ liệu của chunk đầu tiên
            data_to_yield = silence_bytes + data_to_yield
            logger.debug(f"Added {len(silence_bytes)} bytes of silence. First chunk new data length: {len(data_to_yield)}")
        # --- KẾT THÚC THÊM IM LẶNG ---

        # Kiểm tra lại số byte của data_to_yield trước khi yield
        bytes_per_sample_final = target_sample_width
        channels_final = target_channels
        frame_size_final = channels_final * bytes_per_sample_final

        if len(data_to_yield) == 0: # Kiểm tra nếu data_to_yield rỗng
             logger.debug(f"Skipping empty chunk {i+1} after processing.")
             continue

        if len(data_to_yield) % frame_size_final != 0:
             num_frames = len(data_to_yield) // frame_size_final
             if num_frames == 0 and len(data_to_yield) > 0:
                 logger.warning(f"Chunk {i+1} final data size {len(data_to_yield)} is smaller than one frame ({frame_size_final}). Skipping chunk.")
                 continue # Bỏ qua chunk quá nhỏ
             elif num_frames > 0:
                 logger.warning(f"Chunk {i+1} final data size {len(data_to_yield)} for yielding is not perfectly divisible by final frame size ({frame_size_final}). Truncating to {num_frames} frames.")
                 data_to_yield = data_to_yield[:num_frames * frame_size_final] # Cắt bỏ phần thừa

        if len(data_to_yield) == 0: # Kiểm tra lại sau khi truncate
            logger.debug(f"Skipping empty chunk {i+1} after truncation.")
            continue

        # Yield dữ liệu đã (hy vọng) đúng định dạng target (chunk đầu có thêm im lặng)
        yield data_to_yield

        # --- Timing delay ---
        # Tính toán thời gian chờ dựa trên thời lượng *gốc* của chunk (không tính silence thêm vào)
        current_cumulative_duration_sec = (i + 1) * (chunk_duration_ms / 1000.0)
        expected_playback_time = start_time + current_cumulative_duration_sec
        current_time = time.perf_counter()
        playback_delay = expected_playback_time - current_time
        if playback_delay > 0:
            await asyncio.sleep(playback_delay)

    logger.info("Finished yielding all audio chunks.")


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
            # ssl=ssl_context # Bỏ comment nếu kết nối WSS và cần bỏ qua verify SSL
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
                        # Cố gắng lấy kết quả cuối cùng nếu có thể
                        final_transcript = await receive_task
                        logger.info("Received partial transcript after cancellation.")
                    except asyncio.CancelledError:
                        logger.info("Receive task confirmed cancelled after timeout.")
            except asyncio.CancelledError:
                logger.warning("Receive task was cancelled externally.")
                try:
                    final_transcript = await receive_task # Lấy kết quả nếu task bị cancel nhưng vẫn kịp trả về
                except asyncio.CancelledError:
                     pass
            except Exception as e:
                 logger.error(f"Receive task failed with exception: {e}")
                 # Cố gắng lấy kết quả nếu task lỗi nhưng đã xong
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


# --- receive_ws_messages (Giữ nguyên) ---
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


# --- send_ws_audio (Giữ nguyên) ---
async def send_ws_audio(
    websocket,
    file_path: str,
    chunk_duration_ms: int,
    target_sample_rate: int,
    target_channels: int,
    target_sample_width: int
):
    """Streams audio chunks (first one with silence) to the WebSocket server."""
    logger.info("Starting to send audio chunks...")
    sent_chunk_count = 0
    try:
        # Generator giờ đã có logic thêm silence vào chunk đầu
        audio_generator = stream_audio_chunks(
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

# --- Main Execution (Giữ nguyên) ---
async def main():
    parser = argparse.ArgumentParser(description="Test Realtime Endpoints")
    parser.add_argument(
        "protocol", choices=["ws"], help="Protocol to test (ws or rtc)"
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
        logging.getLogger("websockets").setLevel(logging.INFO) # Giữ INFO để tránh quá nhiều log nội bộ
        logging.getLogger("aiortc").setLevel(logging.INFO)   # Giữ INFO

    audio_file_path = Path(args.file)
    if not audio_file_path.is_file():
        logger.error(f"Audio file not found: {args.file}")
        return

    if args.protocol == "ws":
        await test_websocket(args.server, str(audio_file_path), args.model, args.api_key)


if __name__ == "__main__":
    try:
        # Đảm bảo numpy và io được import ở đầu file
        import numpy as np
        import io
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")

    # # --- End of the script ---
    # # Note: The script will exit after the main function completes.
    # model = "erax-ai/EraX-WoW-Turbo-V1.1-CT2" Systran/faster-whisper-large-v3

    #python test_ws.py ws --file generated_429000_long.wav --model erax-ai/EraX-WoW-Turbo-V1.1-CT2 --server http://localhost:8000
