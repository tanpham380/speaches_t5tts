import asyncio
import json
import logging
import time
import ssl
import os

import aiohttp
import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    RTCDataChannel
)
from aiortc.contrib.media import MediaPlayer, MediaRelay
from av import AudioFrame, VideoFrame

# Giả sử file này nằm cùng cấp hoặc trong PYTHONPATH
try:
    from mouble_types.realtime import (
        SessionUpdateEvent,
        ConversationItemCreateEvent,
        ConversationItemContentInputText,
        ConversationItemMessage,
        ResponseCreateEvent,
        PartialSession,
        client_event_type_adapter,
        server_event_type_adapter,
        NotGiven,
        NOT_GIVEN,
        # --- THÊM IMPORT ---
        InputAudioBufferCommitEvent,
        SessionCreatedEvent,
        # --- KẾT THÚC THÊM IMPORT ---
        ResponseTextDeltaEvent,
        ResponseAudioTranscriptDeltaEvent,
        ConversationItemInputAudioTranscriptionCompletedEvent,
        ErrorEvent,
    )
except ImportError as e:
    print(f"Lỗi import mouble_types.realtime: {e}")
    print("Vui lòng đảm bảo file mouble_types/realtime.py có thể truy cập được.")
    exit(1)


# --- Cấu hình ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("realtime_stt_test_client")

SERVER_URL = "http://localhost:8000"
API_ENDPOINT = f"{SERVER_URL}/v1/realtime"
MODEL_ID = "Systran/faster-whisper-large-v3"
AUDIO_FILE_PATH = "generated_429000_long.wav"

if not os.path.exists(AUDIO_FILE_PATH):
    logger.error(f"File âm thanh không tìm thấy tại: {AUDIO_FILE_PATH}")
    exit(1)

ICE_SERVERS = []
RTC_CONFIG = RTCConfiguration(iceServers=[RTCIceServer(**s) for s in ICE_SERVERS])

# --- Biến toàn cục để lưu session ID ---
current_session_id = None

# --- Hàm trợ giúp (giữ nguyên wait_for_connection) ---
async def wait_for_connection(pc: RTCPeerConnection):
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE Connection State is {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            raise ConnectionError("ICE Connection failed")

    while pc.iceConnectionState not in ["completed", "connected"]:
        await asyncio.sleep(0.1)
        if pc.iceConnectionState in ["failed", "closed"]:
             raise ConnectionError(f"ICE Connection failed or closed prematurely ({pc.iceConnectionState})")
    logger.info("ICE Connection Established.")

# --- Logic Test Chính ---
async def run_test():
    global current_session_id # Khai báo để ghi vào biến toàn cục
    pc = RTCPeerConnection(configuration=RTC_CONFIG)
    data_channel = pc.createDataChannel("events")

    logger.info(f"Đang tải file âm thanh từ: {AUDIO_FILE_PATH}")
    player = MediaPlayer(AUDIO_FILE_PATH)
    if player.audio:
        pc.addTrack(player.audio)
    else:
        logger.error("Không tìm thấy audio track trong file.")
        await pc.close()
        return

    received_events = asyncio.Queue()
    client_tasks = set()
    audio_send_complete = asyncio.Event()
    full_transcript = ""

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track {track.kind} received")
        @track.on("ended")
        async def on_ended():
            logger.info(f"Track {track.kind} (từ server) ended")

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.warning(f"Data channel '{channel.label}' created by remote (unexpected)")

    @data_channel.on("open")
    def on_channel_open():
        logger.info(f"Data channel '{data_channel.label}' opened")
        task = asyncio.create_task(send_initial_config(data_channel))
        client_tasks.add(task)
        task.add_done_callback(client_tasks.discard)

    @data_channel.on("close")
    def on_channel_close():
        logger.info(f"Data channel '{data_channel.label}' closed")

    # --- SỬA on_channel_message ĐỂ LƯU session_id ---
    @data_channel.on("message")
    async def on_channel_message(message):
        nonlocal full_transcript
        global current_session_id # Khai báo để ghi
        try:
            event_data = json.loads(message)
            validated_event = server_event_type_adapter.validate_python(event_data)
            logger.info(f"<<< Received event: {validated_event.type}")

            # Lưu session ID nếu nhận được SessionCreatedEvent
            if isinstance(validated_event, SessionCreatedEvent):
                 current_session_id = validated_event.session.id
                 logger.info(f"    Captured Session ID: {current_session_id}")

            # Xử lý các sự kiện phiên âm
            if isinstance(validated_event, ResponseTextDeltaEvent):
                logger.info(f"    Text Delta: '{validated_event.delta}'")
                full_transcript += validated_event.delta
            elif isinstance(validated_event, ResponseAudioTranscriptDeltaEvent):
                 logger.info(f"    Audio Transcript Delta: '{validated_event.delta}'")
                 full_transcript += validated_event.delta
            elif isinstance(validated_event, ConversationItemInputAudioTranscriptionCompletedEvent):
                 logger.info(f"    Transcription COMPLETED for item {validated_event.item_id}")
                 logger.info(f"    Final transcript for item: '{validated_event.transcript}'")
            elif isinstance(validated_event, ErrorEvent):
                 logger.error(f"    SERVER ERROR: {validated_event.error.message} (Type: {validated_event.error.type}, Code: {validated_event.error.code})")

            await received_events.put(validated_event)
        except json.JSONDecodeError:
            logger.error(f"    Failed to decode JSON: {message}")
        except Exception as e:
            logger.error(f"    Failed to validate server event: {e}")

    async def track_media_player(player):
        logger.info("MediaPlayer task started.")
        if player.audio:
            try:
                while True:
                    if player.audio.readyState != "live":
                        logger.info("MediaPlayer audio track is no longer live.")
                        break
                    await asyncio.sleep(0.1)
            except Exception as e:
                 logger.info(f"MediaPlayer task finished or encountered error: {e}")
            finally:
                 logger.info("MediaPlayer task loop finished.")
                 audio_send_complete.set()
        else:
             logger.warning("No audio track in player to track.")
             audio_send_complete.set()

    media_player_task = asyncio.create_task(track_media_player(player))
    client_tasks.add(media_player_task)
    media_player_task.add_done_callback(client_tasks.discard)

    await pc.setLocalDescription(await pc.createOffer())
    offer_sdp = pc.localDescription.sdp
    offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
    logger.info("Created Offer SDP")

    headers = {"Content-Type": "text/plain"}
    params = {"model": MODEL_ID}
    logger.info(f"Sending POST request to {API_ENDPOINT}")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(API_ENDPOINT, params=params, data=offer_sdp, headers=headers, ssl=False) as response:
                response.raise_for_status()
                answer_sdp = await response.text()
                logger.info(f"Received Answer SDP (status: {response.status})")
        except aiohttp.ClientError as e:
            logger.error(f"HTTP Signaling failed: {e}")
            if player.audio: await player.audio.stop()
            await pc.close()
            return
        except Exception as e:
             logger.error(f"An unexpected error occurred during signaling: {e}")
             if player.audio: await player.audio.stop()
             await pc.close()
             return

    answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
    await pc.setRemoteDescription(answer)
    logger.info("Set Remote Description")

    try:
        await wait_for_connection(pc)
    except ConnectionError as e:
         logger.error(f"Connection failed: {e}")
         if player.audio: player.audio.stop()
         await pc.close()
         return

    try:
        while data_channel.readyState != "open":
             logger.debug(f"Waiting for data channel to open (current state: {data_channel.readyState})...")
             await asyncio.sleep(0.2)
        logger.info("Data channel reached 'open' state.")
    except asyncio.TimeoutError:
        logger.error("Timeout waiting for Data Channel to open.")
        if player.audio: player.audio.stop()
        await pc.close()
        return

    logger.info("Streaming audio from file...")
    await audio_send_complete.wait()
    logger.info("Audio file streaming likely finished.")

    # --- GỬI COMMIT EVENT SAU KHI AUDIO GỬI XONG ---
    logger.info("Attempting to send InputAudioBufferCommitEvent...")
    if current_session_id:
         # Giả định item_id của buffer cần commit trùng với session_id
        commit_event = InputAudioBufferCommitEvent(
            type="input_audio_buffer.commit", # <<< THÊM type="input_audio_buffer.commit"
            item_id=current_session_id
        )         
        logger.info(f">>> Sending: {commit_event.type} for item_id={commit_event.item_id}")
        try:
            if data_channel.readyState == "open":
                data_channel.send(commit_event.model_dump_json())
                logger.info("Commit event sent.")
                # Chờ thêm thời gian đáng kể để server xử lý phiên âm sau commit
                logger.info("Waiting after commit for server processing...")
                await asyncio.sleep(10) # Tăng thời gian chờ lên 10 giây
            else:
                logger.warning(f"Cannot send commit event, data channel state is {data_channel.readyState}")
        except Exception as e:
            logger.error(f"Error sending commit event: {e}")
    else:
         logger.warning("Session ID not captured, cannot send commit event.")
    # --- KẾT THÚC GỬI COMMIT EVENT ---

    # Chờ thêm một chút cho các sự kiện cuối cùng (nếu có)
    logger.info("Waiting a final moment for any remaining events...")
    await asyncio.sleep(2)

    logger.info("Finished listening.")
    logger.info("-" * 30)
    logger.info("Full transcript received:")
    logger.info(full_transcript if full_transcript else "[No transcript received]") # In thông báo nếu trống
    logger.info("-" * 30)

    logger.info("Closing connection...")
    for task in client_tasks:
        if not task.done():
            task.cancel()
    try:
        await asyncio.gather(*client_tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass

    if player and player.audio:
        logger.info("Attempting to stop MediaPlayer audio track...")
        try:
            player.audio.stop()
            logger.info("MediaPlayer audio track stopped.")
        except Exception as e:
            logger.warning(f"Error stopping player audio track: {e}")
    else:
        logger.info("MediaPlayer or its audio track is None, skipping stop.")

    await pc.close()
    logger.info("Connection closed.")

async def send_initial_config(channel: RTCDataChannel):
    logger.info("Sending initial configuration...")
    try:
        update_data = PartialSession(
             turn_detection=None, # Gửi None để tắt VAD nếu server cho phép
        )
        fields_to_send = {k: v for k, v in update_data.model_dump().items() if not isinstance(v, NotGiven)}
        if fields_to_send:
            session_update = SessionUpdateEvent(session=PartialSession(**fields_to_send))
            logger.info(f">>> Sending: {session_update.type} with data {fields_to_send}")
            channel.send(session_update.model_dump_json())
        else:
             logger.info("No specific initial config fields set to send.")

        # Gửi ResponseCreateEvent nếu cần thiết để kích hoạt luồng phản hồi/phiên âm
        create_response = ResponseCreateEvent(type="response.create")
        logger.info(f">>> Sending: {create_response.type}")
        channel.send(create_response.model_dump_json())

        await asyncio.sleep(0.5)
        logger.info("Finished sending initial configuration.")
    except Exception as e:
        logger.error(f"Error sending initial configuration: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
         logger.exception(f"An error occurred during the test run: {e}")