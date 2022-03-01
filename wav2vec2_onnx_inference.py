import soundfile as sf
import torch
from transformers import Wav2Vec2Processor
import onnxruntime as rt
import numpy as np

# Improvements: 
# - gpu / cpu flag
# - convert non 16 khz sample rates
# - inference time log

class Wave2Vec2ONNXInference():
    def __init__(self,model_name,onnx_path):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name) 
        #self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        options = rt.SessionOptions()
        options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(onnx_path, options)

    def buffer_to_text(self,audio_buffer):
        if(len(audio_buffer)==0):
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="np", padding=True)

        input_values = inputs.input_values
        onnx_outputs = self.model.run(None, {self.model.get_inputs()[0].name: input_values})[0]
        prediction = np.argmax(onnx_outputs, axis=-1)

        transcription = self.processor.decode(prediction.squeeze().tolist())
        return transcription.lower()

    def file_to_text(self,filename):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)

if __name__ == "__main__":
    print("Model test")
    asr = Wave2Vec2ONNXInference("jonatasgrosman/wav2vec2-large-xlsr-53-german","wav2vec2-large-xlsr-53-german.quant.onnx")
    text = asr.file_to_text("test.wav")
    print(text)