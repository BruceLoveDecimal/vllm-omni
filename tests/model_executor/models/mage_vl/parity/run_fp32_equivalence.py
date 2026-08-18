import json, os, torch
MODEL = "/root/autodl-tmp/models/Mage-VL"
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29607")
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
ctx=set_current_vllm_config(VllmConfig()); ctx.__enter__()
init_distributed_environment(world_size=1, rank=0, distributed_init_method="env://", local_rank=0, backend="nccl")
initialize_model_parallel(tensor_model_parallel_size=1)
from safetensors import safe_open
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm_omni.model_executor.models.mage_vl.vision import MageVLVisionTransformer
from vllm_omni.transformers_utils.configs.mage_vl import MageVLConfig

cfg = MageVLConfig.from_pretrained(MODEL)
torch.set_default_dtype(torch.float32)
with torch.device("cuda"):
    mine = MageVLVisionTransformer(cfg.vision_config, prefix="visual")
mine.eval()
for l in mine.encoder:
    l.self_attn.attn.attn_backend = AttentionBackendEnum.TORCH_SDPA
    l.self_attn.attn.is_flash_attn_backend = False
with open(f"{MODEL}/model.safetensors.index.json") as f: wmap=json.load(f)["weight_map"]
bf={}
for k in [k for k in wmap if k.startswith("model.visual.")]: bf.setdefault(wmap[k],[]).append(k)
def it():
    for fn,keys in bf.items():
        with safe_open(f"{MODEL}/{fn}", framework="pt", device="cpu") as f:
            for k in keys:
                # bf16 on disk -> upcast, so both sides see identical values
                yield k[len("model.visual."):], f.get_tensor(k).cuda().float()
mine.load_weights(it())

from transformers import AutoModelForCausalLM
hf = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32,
                                          attn_implementation="sdpa").eval().cuda()
hvis = hf.model.visual
ref = torch.load("/root/autodl-tmp/mage_ref/img_dog_short.pt", map_location="cuda")
pv, gthw, pp = ref["pixel_values"].cuda().float(), ref["image_grid_thw"].cuda(), ref["patch_positions"].cuda()
with torch.inference_mode():
    a = mine(pv, gthw, pp)
    b = hvis(pv, grid_thw=gthw, patch_positions=pp).last_hidden_state
b = b.squeeze(0) if b.dim()==3 else b
d=(a.float()-b.float())
print("FP32  max_abs=%.4e  rel_l2=%.4e  cos=%.10f" % (
    d.abs().max().item(), (d.norm()/b.float().norm()).item(),
    torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()))
