import torch
import time
import torchvision
import onnx
import onnxruntime as ort
import os


def load_flexvit_model(state_dict_path=None, device='cuda'):
    if state_dict_path:
        state_dict = torch.load(state_dict_path, map_location=device)

    print(state_dict.keys())
    exit()
        
    model_info = get_vit_info(non_pruned_weights=state_dict, core_model=True)
    model = create_vit_general(dim_dict=model_info)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def measure_latency(model, input_size=(32, 3, 224, 224), warmup=10, trials=100):
    dummy_input = torch.randn(*input_size).to(next(model.parameters()).device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Measure latency
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(trials):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_latency = (end_time - start_time) / trials * 1000  # in ms
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


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_flexvit_model(state_dict_path="../pretrained/ViTConfig_(ViTStructureConfig_224_16_12_12_768_3072)_ViTPrebuilt.default_1000_0.0_0.0_(384, 480, 576, 672, 768)_(12, 12, 12, 12, 12)_(1536, 1920, 2304, 2688, 3072)_None.pth", device=device)


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