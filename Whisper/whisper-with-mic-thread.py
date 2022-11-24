from multiprocessing import Process
from threading import Thread
import speech_recognition
import whisper
import io
from pydub import AudioSegment
import tempfile
import os

"""
" An attempt to process in realtime using whisper
" Uses python's multiprocessing library
"""

recognizer = speech_recognition.Recognizer()
model = whisper.load_model('base')
temp_dir = tempfile.mkdtemp()
save_path = os.path.join(temp_dir, "temp.wav")


queue = []


def activateMic():
    counter = 0
    with speech_recognition.Microphone(sample_rate=16000) as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=0.2)
        while True:
            print('Listening' + '.'*(counter+1))
            audio = recognizer.listen(mic)
            audio_data = io.BytesIO(audio.get_wav_data())
            audio_clip = AudioSegment.from_file(audio_data)
            path_name = f'recoding_{counter}'
            audio_clip.export(path_name, format="wav")
            queue.append(path_name)
            counter = (counter+1) % 32
            print(path_name)
            # text = recognizer.recognize_google(audio)

    # except speech_recognition.UnknownValueError():
    #     recognizer = speech_recognition.Recognizer()
    #     continue


def transcribe():
    print('Transcribing')
    while True:
        if len(queue) > 0:
            save_path = queue.pop(0)
            text = model.transcribe(save_path, fp16=False, language='eng')
            text = text['text'].lower()
            print(f"Recognized {text}")
            os.remove(save_path)


def listenAndTranscribe():
    p1 = Thread(target=activateMic, args=[])
    p2 = Thread(target=transcribe, args=[])

    p1.start()
    p2.start()


listenAndTranscribe()
