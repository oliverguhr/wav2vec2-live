import timeit
import soundfile as sf
import torch
from wav2vec2_inference import Wave2Vec2Inference
from wav2vec2_onnx_inference import Wave2Vec2ONNXInference

if __name__ == "__main__":
    torch.set_num_threads(16)
    
    audio_input, samplerate = sf.read("test.wav")
    print("Model test")

    base_model = "oliverguhr/wav2vec2-large-xlsr-53-german-cv9"

    asr_onnx_opt = Wave2Vec2ONNXInference(base_model,"wav2vec2-large-xlsr-53-german-cv9-opt.onnx")
    asr_onnx_quant = Wave2Vec2ONNXInference(base_model,"wav2vec2-large-xlsr-53-german-cv9.quant.onnx")
    asr_onnx = Wave2Vec2ONNXInference(base_model,"wav2vec2-large-xlsr-53-german-cv9.onnx")
    text = asr_onnx.buffer_to_text(audio_input)
    print(text)

    asr = Wave2Vec2Inference(base_model)
    text = asr.buffer_to_text(audio_input)
    print(text)

    iterations = 100
    print(f"Running performance test with {iterations} iterations.")

    seconds = timeit.timeit(lambda: asr.buffer_to_text(audio_input), number=iterations)
    print(f"pytorch inference took {seconds}s for {iterations} iterations. {(seconds/iterations)*1000} ms/iter")

    seconds = timeit.timeit(lambda: asr_onnx_opt.buffer_to_text(audio_input), number=iterations)
    print(f"onnx opt inference took {seconds}s for {iterations} iterations. {(seconds/iterations)*1000} ms/iter")

    seconds = timeit.timeit(lambda: asr_onnx_quant.buffer_to_text(audio_input), number=iterations)
    print(f"onnx quant inference took {seconds}s for {iterations} iterations. {(seconds/iterations)*1000} ms/iter")

    seconds = timeit.timeit(lambda: asr_onnx.buffer_to_text(audio_input), number=iterations)
    print(f"onnx inference took {seconds}s for {iterations} iterations. {(seconds/iterations)*1000} ms/iter")