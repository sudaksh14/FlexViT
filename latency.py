import torch
import time
import torchvision
import onnx
import onnxruntime as ort
import os

from networks import level_delta_utils as delta
from networks import flexvit
import utils
import torch_pruning as tp


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

def export_onnx(model, path, dummy_input=torch.randn((1, 3, 224, 224), dtype=torch.float32), device='cuda'):
    print(f"\nExporting PyTorch model to ONNX: {path}...")
    torch.onnx.export(
        model,                                                                    # The PyTorch model to export
        dummy_input.to(device),                        # A dummy input to trace the model
        path,                                                               # Path where the ONNX model will be saved
        export_params=True,                                                       # Export model parameters (weights)
        opset_version=11,                                          # ONNX opset version (e.g., 11)
        do_constant_folding=True,           # Apply constant folding for optimization
        input_names=['input'],              # Name for the input node in the ONNX graph
        output_names=['output'],            # Name for the output node in the ONNX graph
        dynamic_axes={                      # Define dynamic axes for flexible input sizes
            'input': {0: 'batch_size'},     # Allow variable batch size
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model successfully exported to {path}")

FLEXVIT_CONFIG = flexvit.ViTConfig(
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

if __name__ == "__main__":
    device = utils.get_device()

    model = load_flexvit_model(FLEXVIT_CONFIG, "./pretrained/FlexViT.pt", device=device)

    manager = delta.InMemoryDeltaManager(model, starting_level=0)
    print("Number of Flexible Levels:", manager.max_level() + 1)

    BS = 32

    for i in range(model.max_level() + 1):
        model.set_level_use(i)
        reg_model = model.make_base_copy()
        latency = measure_latency(reg_model, input_size=(BS, 3, 224, 224), warmup=10, trials=100, device=device)
        flops, param = tp.utils.count_ops_and_params(reg_model, torch.randn(1,3,224,224).to(device))
        print(f"🕒 Average Latency (BS={BS}, 224x224) Level {i} latency: {latency:.2f} ms, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}")







    # ----------------------------------------------------VIT-----------------------------------------------------
    # model = ViTForImageClassification.from_pretrained("facebook/deit-base-patch16-224").to(device)
    # model.eval()
    # export_onnx(model, path=os.path.join("./saves/onnx/DeIT_Original.onnx"), device=device)
    # latency_ms = measure_latency(model)
    # print(f"🕒 Average Latency (BS=32, 224x224): {latency_ms:.2f} ms")
    # exit()

    # paths = ["./saves/state_dicts/Vit_b_16_Core_Level_1_state_dict_deit_Iter_Adaptivity_lowlr.pth"] + \
    #         [f"./saves/state_dicts/Vit_b_16_Rebuilt_Level_{i}_state_dict_deit_Iter_Adaptivity_lowlr.pth" for i in range(2,7)]
    
    # paths = ["./saves/state_dicts/Vit_b_16_Rebuilt_Level_5_state_dict_deit_Iter_Adaptivity_lowlr.pth"]
    
    # for state_dict_path in paths:
    #     print(f"Loading model from: {state_dict_path}")
    #     model = load_vit_model(state_dict_path, device=device)
    #     latency_ms = measure_latency(model)
    #     print(f"🕒 Average Latency (BS=32, 224x224): {latency_ms:.2f} ms")

    # for state_dict_path in paths:
    #     print(f"Loading model from: {state_dict_path}")
    #     model = load_vit_model(state_dict_path, device=device)
    #     save_path = os.path.splitext(os.path.basename(state_dict_path))[0]
    #     export_onnx(model, path=os.path.join("./saves/onnx", f"{save_path}.onnx"), device=device)

    # exit()


    # ----------------------------------------------------RESNET-----------------------------------------------------
    # model = torchvision.models.resnet50().to(device)
    # model.eval()
    # latency_ms = measure_latency(model)
    # print(f"🕒 Average Latency (BS=32, 224x224): {latency_ms:.2f} ms")
    # exit()

        
    # paths = ["./saves/state_dicts/Vit_b_16_Core_Level_1_state_dict_Resnet50_Iter_Adaptivity_noSD.pth"] + \
    #         [f"./saves/state_dicts/Vit_b_16_Rebuilt_Level_{i}_state_dict_Resnet50_Iter_Adaptivity_noSD.pth" for i in range(2,7)]

    # for state_dict_path in paths:
    #     print(f"Loading model from: {state_dict_path}")
    #     model = load_resnet_model(state_dict_path, device=device)
    #     save_path = os.path.splitext(os.path.basename(state_dict_path))[0]
    #     export_onnx(model, path=os.path.join("./saves/onnx", f"{save_path}.onnx"), device=device)

    # exit()


    # input_batch_size = 1 # Or 128 if you meant a batch of 128 images
    # input_resolution = 224
    # input_shape = (input_batch_size, 3, input_resolution, input_resolution)

    # Optional: Load an image processor if you were actually processing real images
    # processor = AutoImageProcessor.from_pretrained("facebook/deit-base-patch16-224")
    # print(f"🕒 Average Latency (BS={input_batch_size}, {input_resolution}x{input_resolution}): {latency_ms:.2f} ms")
    
    # for state_dict_path in paths:
    #     print(f"Loading model from: {state_dict_path}")
    #     model = load_resnet_model(state_dict_path, device=device)
    #     latency_ms = measure_latency(model)
    #     print(f"🕒 Average Latency (BS=32, 224x224): {latency_ms:.2f} ms")

    # ----------------------------------------------------VGG16-----------------------------------------------------


    # paths = ["./saves/state_dicts/Vit_b_16_Core_Level_1_state_dict_VGG_Iter_Adaptivity_cifar100_testSD.pth"] + \
    #         [f"./saves/state_dicts/Vit_b_16_Rebuilt_Level_{i}_state_dict_VGG_Iter_Adaptivity_cifar100_testSD.pth" for i in range(2,7)]
    
    # for state_dict_path in paths:
    #     print(f"Loading model from: {state_dict_path}")
    #     model = load_vgg_model(state_dict_path, device=device)
    #     latency_ms = measure_latency(model)
    #     print(f"🕒 Average Latency (BS=32, 224x224): {latency_ms:.2f} ms")