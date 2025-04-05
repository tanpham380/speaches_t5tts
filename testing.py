
# from faster_whisper import WhisperModel

# model_path = "erax-ai/EraX-WoW-Turbo-V1.1-CT2"

# # Convert audio into MONO & 16000 nếu cần thiết
# from pydub import AudioSegment
# # def convert16k(audio_path):
# #     audio = AudioSegment.from_file(audio_path, format="wav")    
# #     audio = audio.split_to_mono()[0]
# #     audio = audio.set_frame_rate(16000)

# #     audio.export("test.wav", format="wav")
# #     return True
    
# # Run on GPU with FP16
# fast_model = WhisperModel(model_path, device="cuda", compute_type="bfloat16", )
# audio ="generated_429000_long.wav"

# # covert_audio = convert16k(audio)

# segments, info = fast_model.transcribe(audio, beam_size=5,
#                                   #word_timestamps=True,
#                                   language="vi",
#                                   temperature=0.0,
#                                   vad_filter=True,
#                                   #vad_parameters=dict(min_silence_duration_ms=2000),
#                                   )

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


# testing.py modification
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio # Import decode_audio
import numpy as np # Import numpy

model_size = "erax-ai/EraX-WoW-Turbo-V1.1-CT2"
audio_file = "generated_429000_long.wav" # Your audio file path

# 1. Decode audio first (like FastAPI)
print(f"Decoding {audio_file}...")
audio_data = decode_audio(audio_file)
print(f"Decoded audio - Shape: {audio_data.shape}, Dtype: {audio_data.dtype}")
# Ensure float32
if audio_data.dtype != np.float32:
    audio_data = audio_data.astype(np.float32)
    print(f"Casted audio to {audio_data.dtype}")

# 2. Load model
print("Loading model...")
# Ensure compute_type matches your confirmed FastAPI setting (bfloat16)
model = WhisperModel(model_size, device="cuda", compute_type="bfloat16")

# 3. Transcribe using the NumPy array and forced parameters
print("Transcribing decoded audio data...")
segments, info = model.transcribe(
    audio_data, # Pass the numpy array
    beam_size=5,
    language="vi",
    temperature=0.0,
    vad_filter=True
    )

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))