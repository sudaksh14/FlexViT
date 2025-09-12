import numpy as np
import sys
import time
import utils

from networks import flexvit
from networks.vit import ViTPrebuilt

import torch_pruning as tp
import torch


def load_flexvit_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sdict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(sdict, strict=False)
    return model


def remap_state_dict_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # class token
        if k == "cls_token":
            new_state_dict["class_token.token"] = v

        # positional embedding
        elif k == "pos_embed":
            new_state_dict["encoder.pos_embedding.embedding"] = v

        # patch embedding
        elif k.startswith("patch_embed.proj."):
            new_key = k.replace("patch_embed.proj.", "conv_proj.")
            new_state_dict[new_key] = v

        # transformer blocks
        elif k.startswith("blocks."):
            parts = k.split(".")
            block_id = int(parts[1])
            sublayer = parts[2]
            prefix = f"encoder.layers.encoder_layer_{block_id}"

            if sublayer == "norm1":
                new_state_dict[f"{prefix}.ln_1.{parts[3]}"] = v

            elif sublayer == "attn":
                if parts[3] == "qkv":
                    # timm packs qkv together, while your model may expect separate?
                    # If same packed format, map directly:
                    new_state_dict[f"{prefix}.self_attention.in_proj_{parts[4]}"] = v
                elif parts[3] == "proj":
                    new_state_dict[f"{prefix}.self_attention.out_proj.{parts[4]}"] = v

            elif sublayer == "norm2":
                new_state_dict[f"{prefix}.ln_2.{parts[3]}"] = v

            elif sublayer == "mlp":
                if parts[3] == "fc1":
                    new_state_dict[f"{prefix}.mlp.0.{parts[4]}"] = v
                elif parts[3] == "fc2":
                    new_state_dict[f"{prefix}.mlp.3.{parts[4]}"] = v

        # final norm
        elif k.startswith("norm."):
            new_state_dict[f"encoder.ln.{k.split('.')[-1]}"] = v

        # classification head
        elif k.startswith("head."):
            new_state_dict[f"heads.head.{k.split('.')[-1]}"] = v

        else:
            print("⚠️ Unmapped key:", k)
    return new_state_dict



def compare_state_dicts(model, checkpoint_path):
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=utils.get_device())
    # ckpt_state = ckpt.get("state_dict", ckpt)  # handle both Lightning & raw dict
    ckpt_state = ckpt["model"] if "model" in ckpt else ckpt

    # Remap checkpoint keys
    ckpt_remapped = remap_state_dict_keys(ckpt_state)

    model_state = model.state_dict()

    # Find matches, missing, and unexpected
    common_keys = set(model_state.keys()) & set(ckpt_remapped.keys())
    missing = set(model_state.keys()) - set(ckpt_remapped.keys())
    unexpected = set(ckpt_remapped.keys()) - set(model_state.keys())

    print(f"✅ Matching keys: {len(common_keys)}")
    print(f"❌ Missing in checkpoint: {len(missing)}")
    print(f"⚠️ Unexpected in checkpoint: {len(unexpected)}")

    if missing:
        print("\n--- Missing keys ---")
        for k in list(missing)[:20]:  # only show first 20
            print(k)

    if unexpected:
        print("\n--- Unexpected keys ---")
        for k in list(unexpected)[:20]:
            print(k)

    return ckpt_remapped


FLEXVIT_CONFIG_NOPREBUILT = flexvit.ViTConfig(
    prebuilt=ViTPrebuilt.noprebuild,
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

FLEXVIT_CONFIG = flexvit.ViTConfig(
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

FLEXVIT_CONFIG_V2 = flexvit.ViTConfig(
    prebuilt=ViTPrebuilt.imagenet1k_v1,
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

FLEXVIT_CONFIG_V3 = flexvit.ViTConfig(
    prebuilt=ViTPrebuilt.Deit_v1,
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

FLEXVIT_CONFIG_V4 = flexvit.ViTConfig(
    prebuilt=ViTPrebuilt.Deit_v3_pretrain_1k,
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

FLEXVIT_CONFIG_V5 = flexvit.ViTConfig(
    prebuilt=ViTPrebuilt.Deit_v3_pretrain_21k,
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

# This script generates the table with the delta file switching timings
if __name__ == "__main__":

    # paths = ["/ivi/xfs/skalra/pretrained/deit_base_patch16_224-b5f2ef4d.pth",
    #          "/ivi/xfs/skalra/pretrained/deit_3_base_224_1k.pth",
    #          "/ivi/xfs/skalra/pretrained/deit_3_base_224_21k.pth"]
    
    # model = FLEXVIT_CONFIG_NOPREBUILT.make_model()
    # model.set_level_use(model.max_level())
    # reg_model = model.make_base_copy()

    # ckpt_deit = compare_state_dicts(reg_model, paths[0])

    # # Compare dicts
    # ckpt_remapped = compare_state_dicts(reg_model, paths[2])
    # ckpt_remapped['encoder.pos_embedding.embedding'] = ckpt_deit['encoder.pos_embedding.embedding']
    # torch.save(ckpt_remapped, "/ivi/xfs/skalra/pretrained/deit_base_v3_21k.pth")

    # missing, unexpected = reg_model.load_state_dict(ckpt_remapped, strict=False)
    # print("Missing:", missing)
    # print("Unexpected:", unexpected)

    # exit()

    device = utils.get_device()
    model1 = FLEXVIT_CONFIG.make_model()
    # model2 = FLEXVIT_CONFIG_V2.make_model()
    # model3 = FLEXVIT_CONFIG_V3.make_model()
    # model4 = FLEXVIT_CONFIG_V4.make_model()
    # model5 = FLEXVIT_CONFIG_V5.make_model()

    _,_,test_loader = utils.load_imagenet(batch_size=512)
    
    paths = ["./pretrained/flexxxxvit_distill.pt",
             "/ivi/xfs/skalra/checkpoints/flexvit_distill_best_model.ckpt",
             "/ivi/xfs/skalra/checkpoints/flexvit_distill_best_model-v1.ckpt"]
        
    print(f"Using FlexViT saved weights at {paths[0]}")
    model = load_flexvit_model(model1, ckpt_path=paths[0], device=device)
    for i in range(model.max_level() + 1):
        model.set_level_use(i)
        reg_model = model.make_base_copy()
        acc = utils.evaluate_model(reg_model, test_loader, device)
        flops, param = tp.utils.count_ops_and_params(reg_model, torch.randn(1,3,224,224).to(device))
        print(f"Level {i} Accuracy: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}")
    
    
    # print("Using Default weights")
    # for i in range(model1.max_level(), model1.max_level() + 1):
    #     model1.set_level_use(i)
    #     reg_model = model1.make_base_copy()
    #     acc = utils.evaluate_model(reg_model, test_loader, device)
    #     flops, param = tp.utils.count_ops_and_params(reg_model, torch.randn(1,3,224,224).to(device))
    #     print(f"Level {i} Accuracy: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}")
       
    # print("Using DeiT (v1) weights")
    # for i in range(model2.max_level(), model2.max_level() + 1):
    #     model2.set_level_use(i)
    #     reg_model = model2.make_base_copy()
    #     acc = utils.evaluate_model(reg_model, test_loader, device)
    #     flops, param = tp.utils.count_ops_and_params(reg_model, torch.randn(1,3,224,224).to(device))
    #     print(f"Level {i} Accuracy: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}")

    # print("Using DeiT v1 weights")
    # for i in range(model3.max_level(), model3.max_level() + 1):
    #     model3.set_level_use(i)
    #     reg_model = model3.make_base_copy()
    #     acc = utils.evaluate_model(reg_model, test_loader, device)
    #     flops, param = tp.utils.count_ops_and_params(reg_model, torch.randn(1,3,224,224).to(device))
    #     print(f"Level {i} Accuracy: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}")
    
    # print("Using DeiT v3 weights with Imagenet-1k pretraining")
    # for i in range(model4.max_level(), model4.max_level() + 1):
    #     model4.set_level_use(i)
    #     reg_model = model4.make_base_copy()
    #     acc = utils.evaluate_model(reg_model, test_loader, device)
    #     flops, param = tp.utils.count_ops_and_params(reg_model, torch.randn(1,3,224,224).to(device))
    #     print(f"Level {i} Accuracy: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}")

    # print("Using DeiT v3 weights with Imagenet-21k pretraining")
    # for i in range(model5.max_level(), model5.max_level() + 1):
    #     model5.set_level_use(i)
    #     reg_model = model5.make_base_copy()
    #     acc = utils.evaluate_model(reg_model, test_loader, device)
    #     flops, param = tp.utils.count_ops_and_params(reg_model, torch.randn(1,3,224,224).to(device))
    #     print(f"Level {i} Accuracy: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}")
    

    
    
    