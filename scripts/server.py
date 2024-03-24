import json
import argparse
import uvicorn
import torch as th
import torch.nn.functional as F
from pyngrok import ngrok
from fastapi import Request, FastAPI
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from core.train.structure_module import StructureModule
from typing import AsyncGenerator


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt")
parser.add_argument("--config", type=str, default="core/train/config/structure_default.yaml")

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

async def structure_generation(data: list[list[bool] | None], y: int) -> AsyncGenerator[list[bool], None]:
    # first element in the list that are none
    indices_to_generate = [i for i, x in enumerate(data) if x is None]
    if len(indices_to_generate) == 0:
        return

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
        next_token = th.multinomial(distribution, num_samples=1).item()

        if j < len(indices_to_generate) - 1:
            next_i = indices_to_generate[j + 1]
            next_sequence = data[i + 1:next_i]
            next_sequence = th.tensor(next_sequence, dtype=th.float, device=model.device).unsqueeze(0)
            next_sequence = model._tube_batch_to_sequence(next_sequence)

            current_sequence = th.cat([th.tensor([[next_token]], device=model.device), next_sequence], dim=1)

        tube = model.idx_to_tube(th.tensor([[next_token]], device=model.device)).squeeze(0).tolist()
        yield f"{json.dumps(tube)}\n"

@app.post("/structure")
async def get_structure(request: Request):
    data = await request.json()
    return StreamingResponse(structure_generation(data["data"], data["y"]))


if __name__ == '__main__':
    print("Running server...")
    uvicorn.run(app=app, host='0.0.0.0', port=8001)
