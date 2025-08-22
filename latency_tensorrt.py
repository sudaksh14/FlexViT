import torch
import time
import torchvision
import os
import argparse
from pathlib import Path
from networks import level_delta_utils as delta
from networks import flexvit
import utils
import torch_pruning as tp
import torch_tensorrt


def load_flexvit_model(config, state_dict_path=None, device='cuda'):
    if state_dict_path:
        state_dict = torch.load(state_dict_path, map_location=device)
        
    model = config.make_model()
    model.load_state_dict(state_dict)
    model.eval()
    return model

def measure_latency(model, input_size=(32, 3, 224, 224), warmup=10, trials=100, device='cuda'):
    model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Measure latency
    torch.cuda.synchronize()
    start_time = time.monotonic_ns()
    with torch.no_grad():
        for _ in range(trials):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    end_time = time.monotonic_ns()

    avg_latency = (end_time - start_time) / trials * 1e-6  # in ms
    return avg_latency

def load_model(model_path: str, device: str):
    """Load a PyTorch model from a .pth or .pt checkpoint."""
    model = torch.load(model_path, map_location=device)
    if isinstance(model, dict) and "state_dict" in model:
        model = model["state_dict"]
    model.eval().to(device)
    return model

def optimize_model(model: torch.nn.Module, input_shape=(1, 3, 224, 224), precision="fp16", workspace_size=1<<22):
    """Optimize the model with Torch-TensorRT."""
    dtype = torch.half if precision == "fp16" else torch.float
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input(input_shape, dtype=dtype)],
        enabled_precisions={dtype},
        workspace_size=workspace_size
    )
    return trt_model

def main():
    parser = argparse.ArgumentParser(description="Optimize PyTorch model with TensorRT")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch model (.pth or .pt)")
    parser.add_argument("--output", type=str, default="trt_model.ts", help="Path to save optimized model")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for optimization")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16"], help="Precision for TensorRT")
    parser.add_argument("--height", type=int, default=224, help="Input height")
    parser.add_argument("--width", type=int, default=224, help="Input width")
    args = parser.parse_args()

    device = "cuda"
    model = load_model(args.model, device)

    input_shape = (args.batch_size, 3, args.height, args.width)
    trt_model = optimize_model(model, input_shape, precision=args.precision)

    torch.jit.save(trt_model, args.output)
    print(f"Optimized model saved to {args.output}")

if __name__ == "__main__":
    device = utils.get_device()
    print(torch.__version__, torch_tensorrt.__version__)
    exit()

    model = load_flexvit_model(FLEXVIT_CONFIG, "./pretrained/FlexViT_5Levels.pt", device=device)

    manager = delta.InMemoryDeltaManager(model, starting_level=0)
    print("Number of Flexible Levels:", manager.max_level() + 1)

    BS = 32

    # USING FLEX MODELS
    for i in range(model.max_level() + 1):
        model.set_level_use(i)
        reg_model = model.make_base_copy()
        latency = measure_latency(reg_model, input_size=(BS, 3, 224, 224), warmup=1, trials=10, device=device)
        flops, param = tp.utils.count_ops_and_params(reg_model, torch.randn(1,3,224,224).to(device))
        print(f"🕒 Average Latency (BS={BS}, 224x224) Level {i} latency: {latency:.2f} ms, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}")