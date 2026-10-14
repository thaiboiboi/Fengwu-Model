import onnxruntime
import onnx
print(onnxruntime.__version__)
print(onnx.__version__)

import onnxruntime as ort

# Check if CUDA is an available provider
providers = ort.get_available_providers()
print(f"Available Providers: {providers}")

if 'CUDAExecutionProvider' in providers:
    print("✅ Success! ONNX Runtime is linked to your GPU.")
else:
    print("❌ Failed. Only CPU is available. Check your cuDNN installation.")

