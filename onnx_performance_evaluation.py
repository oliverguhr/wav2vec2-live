import timeit

from wav2vec2_inference import Wave2Vec2Inference
from wav2vec2_onnx_inference import Wave2Vec2ONNXInference

if __name__ == "__main__":
    print("Model test")
    asr_onnx_opt = Wave2Vec2ONNXInference("jonatasgrosman/wav2vec2-large-xlsr-53-german","wav2vec2-large-xlsr-53-german-opt.onnx")
    asr_onnx_quant = Wave2Vec2ONNXInference("jonatasgrosman/wav2vec2-large-xlsr-53-german","wav2vec2-large-xlsr-53-german.quant.onnx")
    asr_onnx = Wave2Vec2ONNXInference("jonatasgrosman/wav2vec2-large-xlsr-53-german","wav2vec2-large-xlsr-53-german.onnx")
    text = asr_onnx.file_to_text("test.wav")
    print(text)

    asr = Wave2Vec2Inference("jonatasgrosman/wav2vec2-large-xlsr-53-german")
    text = asr.file_to_text("test.wav")
    print(text)

    iterations = 10
    print(f"Running performance test with {iterations} iterations.")

    seconds = timeit.timeit(lambda: asr.file_to_text("test.wav"), number=iterations)
    print(f"pytorch inference took {seconds}s for {iterations} iterations. {(seconds/iterations)*1000} ms/iter")

    seconds = timeit.timeit(lambda: asr_onnx_opt.file_to_text("test.wav"), number=iterations)
    print(f"onnx opt inference took {seconds}s for {iterations} iterations. {(seconds/iterations)*1000} ms/iter")

    seconds = timeit.timeit(lambda: asr_onnx_quant.file_to_text("test.wav"), number=iterations)
    print(f"onnx quant inference took {seconds}s for {iterations} iterations. {(seconds/iterations)*1000} ms/iter")

    seconds = timeit.timeit(lambda: asr_onnx.file_to_text("test.wav"), number=iterations)
    print(f"onnx inference took {seconds}s for {iterations} iterations. {(seconds/iterations)*1000} ms/iter")