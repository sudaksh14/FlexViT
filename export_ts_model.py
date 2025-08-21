import os, copy, random, numpy as np, torch
from torch import nn
from networks import flexvit

# ---------- Determinism ----------
torch.manual_seed(0); np.random.seed(0); random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

# ---------- Helpers ----------
def to_static_half_if_you_want(m: nn.Module, use_fp16: bool, device: torch.device):
    m = m.to(device)
    if use_fp16:
        m = m.half()
    m.eval()
    return m

def script_or_trace_static(model: nn.Module, example: torch.Tensor):
    # Try script first (handles control flow); fall back to strict trace
    try:
        scripted = torch.jit.script(model)
    except Exception:
        scripted = torch.jit.trace(model, example, strict=True)
    # Freeze to stabilize graph & constant fold
    scripted = torch.jit.freeze(scripted.eval())
    return scripted

FLEXVIT_CONFIG = flexvit.ViTConfig(
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    USE_FP16 = False  # set True if you want FP16 end-to-end

    # --- Build & load base model ---
    model = FLEXVIT_CONFIG.make_model()
    state = torch.load("./pretrained/FlexViT_5Levels.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # fixed, static input shape for export
    B, C, H, W = 1, 3, 224, 224
    example_input = torch.randn(B, C, H, W, device=device)
    if USE_FP16: example_input = example_input.half()

    os.makedirs("./pretrained/export_levels", exist_ok=True)

    # --- Export each level as a fully static module ---
    with torch.no_grad():
        max_lv = model.max_level()
        for i in range(max_lv + 1):
            # Work on an isolated copy so no mutable state leaks across levels
            reg_model = copy.deepcopy(model)
            reg_model.set_level_use(i)          # lock level before TS export
            reg_model = to_static_half_if_you_want(reg_model, USE_FP16, device)

            # (Optional) remove dropout if any slipped through
            for name, child in list(reg_model.named_modules()):
                if isinstance(child, torch.nn.Dropout):
                    setattr(reg_model, name.split('.')[-1], torch.nn.Identity())

            ts = script_or_trace_static(reg_model, example_input)
            out_path = f"./pretrained/export_levels/FlexViT_level_{i}.pt"
            ts.save(out_path)
            print(f"Saved level {i} -> {out_path}")
