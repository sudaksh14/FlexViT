import os, time
import torch

try:
    import torch_tensorrt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("⚠️ torch_tensorrt not installed — will only run TorchScript models")

def load_model(path: str, use_trt: bool = False, device="cuda"):
    """
    Loads either TorchScript or TRT-optimized TorchScript model.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    model = torch.jit.load(path, map_location=device).eval()
    if use_trt and not TRT_AVAILABLE:
        raise RuntimeError("TensorRT not available in this env")

    return model.to(device)

def run_inference(model, input_tensor, n_runs=50):
    """
    Warmup + measure avg latency.
    """
    with torch.no_grad():
        # warmup
        for _ in range(5):
            _ = model(input_tensor)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_runs):
            _ = model(input_tensor)
        torch.cuda.synchronize()
        end = time.time()

    avg_ms = (end - start) * 1000 / n_runs
    return avg_ms

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    BS, C, H, W = 1, 3, 224, 224
    x = torch.randn(BS, C, H, W, device=device)

    # Example: load and run all levels
    num_levels = 5   # adjust to match your model
    for i in range(num_levels + 1):
        ts_path = f"./pretrained/FlexViT_level_{i}.pt"
        trt_path = f"./pretrained/FlexViT_level_{i}_trt.pt"

        print(f"\n=== Level {i} ===")

        # TorchScript baseline
        ts_model = load_model(ts_path, use_trt=False, device=device)
        ts_latency = run_inference(ts_model, x)
        print(f"TorchScript latency: {ts_latency:.2f} ms")

        # TRT optimized (if available)
        if TRT_AVAILABLE and os.path.exists(trt_path):
            trt_model = load_model(trt_path, use_trt=True, device=device)
            trt_latency = run_inference(trt_model, x)
            print(f"TensorRT latency:   {trt_latency:.2f} ms")
        else:
            print("No TRT engine found for this level.")
