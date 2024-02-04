from flask import Flask, render_template
from flask_sockets import Sockets
from wav2vec2_inference import Wave2Vec2Inference
import threading
from queue import Queue
import numpy as np
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
import base64

app = Flask(__name__)
sockets = Sockets(app)

class LiveWav2Vec2:
    exit_event = threading.Event()

    def __init__(self, model_name):
        self.model_name = model_name

    def stop(self):
        """stop the asr process"""
        LiveWav2Vec2.exit_event.set()
        print("asr stopped")

    def start(self):
        """start the asr process"""
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()

        self.asr_process = threading.Thread(target=LiveWav2Vec2.asr_process, args=(
            self.model_name, self.asr_input_queue, self.asr_output_queue,))
        self.asr_process.start()

    @staticmethod
    def asr_process(model_name, in_queue, output_queue):
        wave2vec_asr = Wave2Vec2Inference(model_name, use_lm_if_possible=True)

        print("\nlistening to your voice\n")
        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break
            float64_buffer = np.frombuffer(audio_frames, dtype=np.int16) / 32767
            text, confidence = wave2vec_asr.buffer_to_text(float64_buffer)
            text = text.lower()
            sample_length = len(audio_frames) / 16000
            
            if text != "":
                output_queue.put([text, sample_length, confidence])
        

    def get_last_text(self):
        """returns the text, sample length and inference time in seconds."""
        return self.asr_output_queue.get()

@app.route('/')
def index():
    return render_template('index.html')

@sockets.route('/audio_stream')
def audio_stream(ws):
    live_wav2vec2 = LiveWav2Vec2("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    live_wav2vec2.start()

    try:
        while not ws.closed:
            message = ws.receive()
            message = base64.b64decode(message)
            if message:
                live_wav2vec2.asr_input_queue.put(message)
            if message:
                text, sample_length, confidence = live_wav2vec2.get_last_text()
                print(f"{sample_length:.3f}s\t{confidence}\t{text}")
                ws.send(text)
    except KeyboardInterrupt:
        live_wav2vec2.stop()
        exit()

if __name__ == "__main__":
    server = pywsgi.WSGIServer(('127.0.0.1', 5010), app, handler_class=WebSocketHandler)
    print("Server running at http://127.0.0.1:5010/")
    server.serve_forever()