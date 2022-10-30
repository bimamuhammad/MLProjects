import speech_recognition
import whisper
import io
from pydub import AudioSegment
import tempfile
import os

recognizer = speech_recognition.Recognizer()
model = whisper.load_model('base')
temp_dir = tempfile.mkdtemp()
save_path = os.path.join(temp_dir, "temp.wav")

# while True:
#     try:
with speech_recognition.Microphone(sample_rate=16000) as mic:
    recognizer.adjust_for_ambient_noise(mic, duration=0.2)
    print("Speak ........")
    while True:
        audio = recognizer.listen(mic)
        audio_data = io.BytesIO(audio.get_wav_data())
        audio_clip = AudioSegment.from_file(audio_data)
        audio_clip.export(save_path, format="wav")
        # text = recognizer.recognize_google(audio)
        text = model.transcribe(save_path, fp16=False, language='hausa')
        text = text['text'].lower()

        print(f"Recognized {text}")
    # except speech_recognition.UnknownValueError():
    #     recognizer = speech_recognition.Recognizer()
    #     continue
