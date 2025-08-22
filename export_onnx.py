import torch
from networks import flexvit

FP16 = True
 
FLEXVIT_CONFIG = flexvit.ViTConfig(
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FLEXVIT_CONFIG.make_model()
model.load_state_dict(torch.load("./pretrained/FlexViT_5Levels.pt", map_location=device))
model.eval()

if FP16:
    example_input = torch.randn(1, 3, 224, 224).half().to(device)
else:
    example_input = torch.randn(1, 3, 224, 224).to(device)

for i in range(model.max_level() + 1):
    model.set_level_use(i)
    reg_model = model.make_base_copy().to(device)
    if FP16:
        reg_model.half()
    

    torch.onnx.export(
        reg_model, example_input,
        f"./pretrained/FlexViT_level_{i}.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        do_constant_folding=True
    )
    print(f"✅ Exported FlexViT Level {i} to ONNX")
