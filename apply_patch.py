from pathlib import Path
import torch

p = Path("./flex_modules/selfattention.py")
code = p.read_text()

# Ensure torch is imported in the file; it's already present.
# 1) Insert target_nheads into UpDelta in export_level_delta and modify apply_level_delta_up to handle full replace vs delta concat.

code = code.replace(
    "delta_up = (target_inw, target_inb, target_ow, out_bias)\n        delta_down = (target_token, self.heads[target_level])",
    "delta_up = (target_inw, target_inb, target_ow, out_bias, torch.tensor([self.heads[target_level]], dtype=torch.int64, device=w_in_full.device))\n        delta_down = (target_token, self.heads[target_level])"
)

code = code.replace(
    "def apply_level_delta_up(b: nn.MultiheadAttention, level_delta: UpDelta[tuple[torch.Tensor, ...]]) -> None:\n        target_inw, target_inb, target_ow, out_bias = level_delta.delta",
    "def apply_level_delta_up(b: nn.MultiheadAttention, level_delta: UpDelta[tuple[torch.Tensor, ...]]) -> None:\n        target_inw, target_inb, target_ow, out_bias, target_nheads_tensor = level_delta.delta\n\n        # target_nheads (int)\n        target_nheads = int(target_nheads_tensor.item())"
)

old_block = """        # nheads = b.num_heads + target_heads_inw.shape[1]
        # nhead_dim = b.head_dim + curr_bottom_inw.shape[2]
        print(b.embed_dim)
        print(target_inw.shape)
        nembed_dim = b.embed_dim + target_inw.shape[1]
        print(nembed_dim)

        t = b.in_proj_weight.data
        print(t.shape)
        print(target_inw.shape)
        t = torch.cat([t, target_inw.to(t)], dim=1)
        print(t.shape)
        b.in_proj_weight.data = t.detach()

        t = b.in_proj_bias.data
        print(t.shape)
        t = torch.cat([t, target_inb.to(t)], dim=0)
        print(t.shape)
        b.in_proj_bias.data = t.detach()

        t = b.out_proj.weight.data
        t = torch.cat([t, target_ow.to(t)], dim=1)
        print(t.shape)
        b.out_proj.weight.data = t.detach()

        t = torch.cat([b.out_proj.bias.data, out_bias.to(t)])
        b.out_proj.bias.data = t.detach()

        b.embed_dim = nembed_dim"""
new_block = """        # Determine if UpDelta is a full-replace (coming from level-0) or incremental deltas.
        # Full replace: target_inw has shape (3*T, T) and target_ow is (T, T)
        if target_inw is None:
            return

        if (target_inw.dim() == 2 and target_ow.dim() == 2 and
            target_inw.shape[0] == 3 * target_inw.shape[1] and
            target_ow.shape[0] == target_ow.shape[1] == target_inw.shape[1]):
            # Full replace: just set tensors directly
            T = target_inw.shape[1]
            b.in_proj_weight.data = target_inw.detach().clone()
            b.in_proj_bias.data = target_inb.detach().clone()
            b.out_proj.weight.data = target_ow.detach().clone()
            b.out_proj.bias.data = out_bias.detach().clone()

            b.embed_dim = T
            b.num_heads = target_nheads
            b.head_dim = T // target_nheads
            return

        # Otherwise treat as incremental delta: append blocks (right columns, bottom rows, bottom-right)
        old_in = b.in_proj_weight.data
        # Append columns (to the right)
        new_top = torch.cat([old_in, target_inw.to(old_in)], dim=1)
        # If target also contains bottom rows, append them
        if target_inw.shape[0] > old_in.shape[0]:
            # bottom block provided: append rows from appropriate slice if possible
            bottom = target_inw.to(old_in)
            new_in = torch.cat([new_top, bottom], dim=0)
        else:
            new_in = new_top

        b.in_proj_weight.data = new_in.detach()

        # Bias
        b.in_proj_bias.data = torch.cat([b.in_proj_bias.data, target_inb.to(b.in_proj_bias.data)]).detach()

        # out proj weight: append columns then rows similarly
        old_out = b.out_proj.weight.data
        new_out = torch.cat([old_out, target_ow.to(old_out)], dim=1)
        if target_ow.shape[0] > old_out.shape[0]:
            new_out = torch.cat([new_out, target_ow.to(new_out)], dim=0)
        b.out_proj.weight.data = new_out.detach()

        b.out_proj.bias.data = torch.cat([b.out_proj.bias.data, out_bias.to(b.out_proj.bias.data)]).detach()

        # Update dims: set embed_dim and num_heads
        nembed_dim = b.in_proj_weight.shape[1]
        b.embed_dim = nembed_dim
        b.num_heads = target_nheads
        b.head_dim = nembed_dim // target_nheads"""

if old_block in code:
    code = code.replace(old_block, new_block)
else:
    # fallback: try to find the print(b.embed_dim) anchor and replace following lines until b.embed_dim assignment
    import re
    code = re.sub(r"print\\(b.embed_dim\\)[\\s\\S]*?b\\.embed_dim = nembed_dim", new_block, code, count=1)

p.write_text(code)
print("Patched file saved.")
