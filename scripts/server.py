import json
import os
import argparse
import uvicorn
import yaml
import torch as th
import torch.nn.functional as F
from itertools import product, chain
from functools import partial
from pyngrok import ngrok
from fastapi import Request, FastAPI
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from core.train.structure_module import StructureModule
from core.train.color_module import ColorModule
from typing import Generator
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--only_color", action="store_true")
parser.add_argument("--only_structure", action="store_true")
parser.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "epoch=0-step=6000.ckpt"))
parser.add_argument("--ckpt_color", type=str, default=os.path.join("checkpoints_color", "last.ckpt"))
parser.add_argument("--config", type=str, default=os.path.join("core", "train", "config", "structure_default.yaml"))
parser.add_argument("--config_color", type=str, default=os.path.join("core", "train", "config", "color_default.yaml"))
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--ngrok", action="store_true")

args = parser.parse_args()

assert not (args.only_color and args.only_structure), "Cannot specify both only_color and only_structure"

if not args.only_color:
    print("Loading structure model...")
    model = StructureModule.from_conf(args.config)
    ckpt = th.load(args.ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model.cuda()
    model.eval()

    del ckpt

if not args.only_structure:
    with open(args.config_color, "r") as file:
        conf = yaml.safe_load(file)

        # First flatten conf
        defaults = {
            key: value for key, value in chain(*[category.items() for category in conf.values() if isinstance(category, dict)])
        }

        # Then update any args that are None with the defaults
        for key, value in defaults.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

    model_kwargs = vars(args)

    print("Loading color model...")
    color_model = ColorModule.from_conf(model_kwargs.pop("config_color"), **model_kwargs)
    ckpt = th.load(args.ckpt_color)
    color_model.load_state_dict(ckpt["state_dict"])
    color_model.cuda()
    color_model.eval()

    del ckpt

print("Setting up API...")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if args.ngrok:
    print("Setting up ngrok tunnel...")
    http_tunnel = ngrok.connect(args.port, "http")
    print(http_tunnel)

def greedy(probs: th.Tensor) -> th.Tensor:
    return th.argmax(probs, dim=-1)

def sample(probs: th.Tensor) -> th.Tensor:
    return th.multinomial(probs, num_samples=1)

def topk(k: int, probs: th.Tensor) -> th.Tensor:
    topk_probs, topk_indices = th.topk(probs, k, dim=-1)
    topk_probs /= topk_probs.sum(dim=-1, keepdim=True)

    samples = th.multinomial(topk_probs, num_samples=1)

    return topk_indices.gather(-1, samples)

def nucleus(p: float, probs: th.Tensor) -> th.Tensor:
    sorted_probs, sorted_indices = th.sort(probs, descending=True, dim=-1)
    sorted_probs = th.cumsum(sorted_probs, dim=-1)
    sorted_indices = sorted_indices[sorted_probs < p]
    if sorted_indices.numel() == 0:
        sorted_indices = th.argmax(probs, dim=-1, keepdim=True)  # if we get 0 probs in the nucleus, just take the max

    samples = th.multinomial(probs[sorted_indices], num_samples=1)

    return sorted_indices[samples]

sampling_strategies = {
    "greedy": greedy,
    "sample": sample,
    "topk": topk,
    "nucleus": nucleus
}

def structure_generation(data: list[list[bool | None]], y: int, sampling_strategy: dict) -> Generator[str, None, None]:
    # first element in the list that are none
    indices_to_generate = [i for i, x in enumerate(data) if None in x]
    if len(indices_to_generate) == 0:
        return
    
    strategy_name = sampling_strategy.get("strategy", "greedy")
    
    if strategy_name == "topk":
        sampling_fn = partial(sampling_strategies[strategy_name], sampling_strategy["k"])
    elif strategy_name == "nucleus":
        sampling_fn = partial(sampling_strategies[strategy_name], sampling_strategy["p"])
    else:
        sampling_fn = sampling_strategies[strategy_name]

    bos_tube = th.full((1, model.tube_length), -1, dtype=th.float, device=model.device)
    current_sequence = data[:indices_to_generate[0]]  # add the known data
    current_sequence = th.tensor(current_sequence, dtype=th.float, device=model.device).unsqueeze(0)
    current_sequence = model._tube_batch_to_sequence(current_sequence, bos_tube)

    y_indices = th.tensor([y], device=model.device)
    cache = None

    for j, i in enumerate(indices_to_generate):
        output = model(current_sequence, y_indices, use_cache=True, past_key_values=cache)
        cache = output.past_key_values
        distribution = F.softmax(output.logits[0, -1], dim=-1)
        next_token = sampling_fn(distribution)
        next_token = th.tensor([[next_token.item()]], device=model.device)

        original_tube = data[i]
        tube = model.idx_to_tube(next_token).squeeze(0).tolist()

        yield f"{json.dumps(tube)}\n"

        # replace the None values with the generated tube
        tube = [original_tube[k] if original_tube[k] is not None else tube[k] for k in range(len(tube))]
        tube = th.tensor(tube, dtype=th.float, device=model.device).unsqueeze(0)
        
        if th.all(tube == -1):  # reached the end of the sequence
            break

        next_token = model.tube_to_idx(tube)

        if j < len(indices_to_generate) - 1:
            next_i = indices_to_generate[j + 1]
            next_sequence = data[i + 1:next_i]
            next_sequence = th.tensor(next_sequence, dtype=th.float, device=model.device).unsqueeze(0)
            next_sequence = model._tube_batch_to_sequence(next_sequence)

            current_sequence = th.cat([next_token, next_sequence], dim=1)

@app.post("/structure")
async def get_structure(request: Request):
    data = await request.json()
    return StreamingResponse(structure_generation(data["data"], data["y"], data["sampling"]))


def color_generation(data: list[list[list[list[int | None]]]], mask: list[list[list[list[bool]]]], y: int, steps: int, strength: float) -> Generator[str, None, None]:
    b_len, y_len, z_len, x_len = len(data), len(data[0]), len(data[0][0]), len(data[0][0][0])
    
    assert b_len == len(mask)
    assert y_len == len(mask[0])
    assert z_len == len(mask[0][0])
    assert x_len == len(mask[0][0][0])

    mask = th.tensor(mask, dtype=th.bool, device=color_model.device).view(b_len, 1, y_len, z_len, x_len)  # (b, c, y, z, x)
    context_tensor = th.tensor(data, dtype=th.uint8, device=color_model.device).view(b_len, 1, y_len, z_len, x_len)

    control = {
        "structure": context_tensor > 0,
        "y_index": y,
    }

    for step in tqdm(color_model(1, steps, "heun_sde_inpaint", ctrl=control, context=context_tensor, mask=mask, inpaint_strength=strength, clamp=True, use_ema=True), total=steps, desc="Color generation", leave=False):
        generated_region = step[mask].tolist()
        generated_coords = th.nonzero(mask).tolist()

        generated_block_coords = [(coords, block_id) for coords, block_id in zip(generated_coords, generated_region)]

        yield f"{json.dumps(generated_block_coords)}\n"

@app.post("/color")
async def get_color(request: Request):
    data = await request.json()
    return StreamingResponse(color_generation(data["data"], data["mask"], data["y"], data["steps"], data["strength"]))


if __name__ == '__main__':
    print("Running server...")
    uvicorn.run(app=app, host='0.0.0.0', port=8001)
