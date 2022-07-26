import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor, pipeline

# Improvements: 
# - gpu / cpu flag
# - convert non 16 khz sample rates
# - inference time log

class Wave2Vec2Inference():
    def __init__(self,model_name, hotwords=[]):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.hotwords = hotwords

    def buffer_to_text(self,audio_buffer):
        if(len(audio_buffer)==0):
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            result = self.model(inputs.input_values)
            logits = result.logits

        if hasattr(self.processor, 'decoder'):
            transcription = \
                self.processor.decode(logits[0].cpu().numpy(),                                      
                                      hotwords=self.hotwords,
                                      #hotword_weight=self.hotword_weight,                                      
                                   )            
            transcription = transcription.text
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription    

    def file_to_text(self,filename):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)
    
if __name__ == "__main__":
    print("Model test")
    asr = Wave2Vec2Inference("oliverguhr/wav2vec2-large-xlsr-53-german-cv9")
    text = asr.file_to_text("test.wav")
    print(text)