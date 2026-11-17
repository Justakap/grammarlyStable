import os
import sys
import json
import math
import shutil

import torch
from safetensors.torch import load_file, save_file

SRC = os.getenv("SRC_ADAPTER", "../qwen_4b_hinglish_lora")
DST = os.getenv("DST_ADAPTER", "")
RANK = int(os.getenv("RANK", "0"))
ALPHA = int(os.getenv("ALPHA", str(2 * RANK)))
SEED = int(os.getenv("SEED", "42"))

if not DST or RANK <= 0:
    sys.exit("Set DST_ADAPTER and RANK")

with open(os.path.join(SRC, "adapter_config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
if cfg.get("rank_pattern") or cfg.get("alpha_pattern") or cfg.get("use_rslora") or cfg.get("use_dora"):
    sys.exit("Only plain LoRA adapters without per-module rank/alpha patterns are supported")

old_r, old_alpha = int(cfg["r"]), int(cfg["lora_alpha"])
old_scale, new_scale = old_alpha / old_r, ALPHA / RANK
state = load_file(os.path.join(SRC, "adapter_model.safetensors"))
torch.manual_seed(SEED)

out, errors, energies = {}, [], []
for key in sorted(state):
    if key.endswith("lora_B.weight"):
        continue
    if not key.endswith("lora_A.weight"):
        out[key] = state[key]
        continue
    key_b = key[: -len("lora_A.weight")] + "lora_B.weight"
    a = state[key].float()
    b = state[key_b].float() * (old_scale / new_scale)
    if RANK >= old_r:
        extra = RANK - old_r
        a_extra = torch.empty(extra, a.shape[1])
        if extra:
            torch.nn.init.kaiming_uniform_(a_extra, a=math.sqrt(5))
        a_new = torch.cat([a, a_extra], dim=0)
        b_new = torch.cat([b, torch.zeros(b.shape[0], extra)], dim=1)
    else:
        q_b, r_b = torch.linalg.qr(b)
        q_a, r_a = torch.linalg.qr(a.T)
        u, s, vh = torch.linalg.svd(r_b @ r_a.T)
        root = s[:RANK].sqrt()
        b_new = q_b @ (u[:, :RANK] * root)
        a_new = (root[:, None] * vh[:RANK]) @ q_a.T
        energies.append(float((s[:RANK] ** 2).sum() / (s ** 2).sum().clamp_min(1e-30)))
    probe = torch.randn(a.shape[1], 4)
    ref = old_scale * (state[key_b].float() @ (state[key].float() @ probe))
    new = new_scale * (b_new @ (a_new @ probe))
    errors.append(float((new - ref).norm() / ref.norm().clamp_min(1e-12)))
    out[key] = a_new.to(state[key].dtype).contiguous()
    out[key_b] = b_new.to(state[key_b].dtype).contiguous()

pairs = len(errors)
print(f"Resized {pairs} LoRA modules: rank {old_r} -> {RANK}, alpha {old_alpha} -> {ALPHA}")
print(f"Output difference vs original adapter: mean {sum(errors) / pairs:.2e} | max {max(errors):.2e}")
if RANK >= old_r and max(errors) > 1e-3:
    sys.exit("Expanded adapter does not reproduce the original; stopping")
if energies:
    print(f"Share of each update kept at rank {RANK}: mean {sum(energies) / len(energies):.4f} | "
          f"min {min(energies):.4f}")

os.makedirs(DST, exist_ok=True)
save_file(out, os.path.join(DST, "adapter_model.safetensors"), metadata={"format": "pt"})
cfg["r"], cfg["lora_alpha"] = RANK, ALPHA
with open(os.path.join(DST, "adapter_config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
for name in os.listdir(SRC):
    if name not in ("adapter_model.safetensors", "adapter_config.json") and os.path.isfile(os.path.join(SRC, name)):
        shutil.copy2(os.path.join(SRC, name), os.path.join(DST, name))
print(f"Saved {DST}")
