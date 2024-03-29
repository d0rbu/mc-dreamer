import json
import os
import argparse
import uvicorn
import torch as th
import torch.nn.functional as F
from functools import partial
from pyngrok import ngrok
from fastapi import Request, FastAPI
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from core.train.structure_module import StructureModule
from typing import AsyncGenerator


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "epoch=0-step=6000.ckpt"))
parser.add_argument("--config", type=str, default=os.path.join("core", "train", "config", "structure_default.yaml"))

args = parser.parse_args()

print("Loading model...")
model = StructureModule.from_conf(args.config)
ckpt = th.load(args.ckpt)
model.load_state_dict(ckpt["state_dict"])
model.eval()

del ckpt

print("Setting up API...")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

print("Setting up ngrok tunnel...")
http_tunnel = ngrok.connect("8001", "http")
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

    samples = th.multinomial(probs[sorted_indices], num_samples=1)

    return sorted_indices[samples]

sampling_strategies = {
    "greedy": greedy,
    "sample": sample,
    "topk": topk,
    "nucleus": nucleus
}

async def structure_generation(data: list[list[bool | None]], y: int, sampling_strategy: dict) -> AsyncGenerator[list[bool], None]:
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


if __name__ == '__main__':
    print("Running server...")
    uvicorn.run(app=app, host='0.0.0.0', port=8001)
