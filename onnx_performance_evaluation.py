import timeit
import soundfile as sf
from wav2vec2_inference import Wave2Vec2Inference
from wav2vec2_onnx_inference import Wave2Vec2ONNXInference

if __name__ == "__main__":
    audio_input, samplerate = sf.read("test.wav")
    print("Model test")
    asr_onnx_opt = Wave2Vec2ONNXInference("jonatasgrosman/wav2vec2-large-xlsr-53-german","wav2vec2-large-xlsr-53-german-opt.onnx")
    asr_onnx_quant = Wave2Vec2ONNXInference("jonatasgrosman/wav2vec2-large-xlsr-53-german","wav2vec2-large-xlsr-53-german.quant.onnx")
    asr_onnx = Wave2Vec2ONNXInference("jonatasgrosman/wav2vec2-large-xlsr-53-german","wav2vec2-large-xlsr-53-german.onnx")
    text = asr_onnx.buffer_to_text(audio_input)
    print(text)

    asr = Wave2Vec2Inference("jonatasgrosman/wav2vec2-large-xlsr-53-german")
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