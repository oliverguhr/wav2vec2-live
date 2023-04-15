import pyaudio
import webrtcvad
from wav2vec2_inference import Wave2Vec2Inference
import numpy as np
import threading
import time
from sys import exit
from queue import  Queue


class LiveWav2Vec2:
    exit_event = threading.Event()
    def __init__(self, model_name, device_name="default"):
        self.model_name = model_name
        self.device_name = device_name

    def stop(self):
        """stop the asr process"""
        LiveWav2Vec2.exit_event.set()
        self.asr_input_queue.put("close")
        print("asr stopped")

    def start(self):
        """start the asr process"""
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()
        self.asr_process = threading.Thread(target=LiveWav2Vec2.asr_process, args=(
            self.model_name, self.asr_input_queue, self.asr_output_queue,))
        self.asr_process.start()
        time.sleep(5)  # start vad after asr model is loaded
        self.vad_process = threading.Thread(target=LiveWav2Vec2.vad_process, args=(
            self.device_name, self.asr_input_queue,))
        self.vad_process.start()

    @staticmethod
    def vad_process(device_name, asr_input_queue):
        vad = webrtcvad.Vad()
        vad.set_mode(1)

        audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        # A frame must be either 10, 20, or 30 ms in duration for webrtcvad
        FRAME_DURATION = 30
        CHUNK = int(RATE * FRAME_DURATION / 1000)

        microphones = LiveWav2Vec2.list_microphones(audio)
        selected_input_device_id = LiveWav2Vec2.get_input_device_id(
            device_name, microphones)

        stream = audio.open(input_device_index=selected_input_device_id,
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        frames = b''
        while True:
            if LiveWav2Vec2.exit_event.is_set():
                break
            frame = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)
            if is_speech:
                frames += frame
            else:
                if len(frames) > 1:
                    asr_input_queue.put(frames)
                frames = b''
        stream.stop_stream()
        stream.close()
        audio.terminate()

    @staticmethod
    def asr_process(model_name, in_queue, output_queue):
        wave2vec_asr = Wave2Vec2Inference(model_name, use_lm_if_possible=True)

        print("\nlistening to your voice\n")
        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break

            float64_buffer = np.frombuffer(
                audio_frames, dtype=np.int16) / 32767
            start = time.perf_counter()
            text, confidence = wave2vec_asr.buffer_to_text(float64_buffer)
            text = text.lower()
            inference_time = time.perf_counter()-start
            sample_length = len(float64_buffer) / 16000  # length in sec
            if text != "":
                output_queue.put([text,sample_length,inference_time,confidence])

    @staticmethod
    def get_input_device_id(device_name, microphones):
        for device in microphones:
            if device_name in device[1]:
                return device[0]

    @staticmethod
    def list_microphones(pyaudio_instance):
        info = pyaudio_instance.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        result = []
        for i in range(0, numdevices):
            if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = pyaudio_instance.get_device_info_by_host_api_device_index(
                    0, i).get('name')
                result += [[i, name]]
        return result

    def get_last_text(self):
        """returns the text, sample length and inference time in seconds."""
        return self.asr_output_queue.get()

if __name__ == "__main__":
    print("Live ASR")

    asr = LiveWav2Vec2("oliverguhr/wav2vec2-large-xlsr-53-german-cv9")

    asr.start()

    try:
        while True:
            text, sample_length, inference_time, confidence = asr.get_last_text()
            print(f"{sample_length:.3f}s\t{inference_time:.3f}s\t{confidence}\t{text}")

    except KeyboardInterrupt:
        asr.stop()
        exit()
