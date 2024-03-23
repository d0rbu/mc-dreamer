import json
import torch as th
import torch.nn.functional as F
from fastapi import Request, FastAPI
from fastapi.responses import StreamingResponse
from core.train.structure_module import StructureModule
from typing import AsyncGenerator


CHECKPOINT_PATH = "model.ckpt"

model = StructureModule.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()

app = FastAPI()

async def structure_generation(data: list[list[bool] | None], y: int) -> AsyncGenerator[list[bool], None, None]:
    # first element in the list that are none
    indices_to_generate = [i for i, x in enumerate(data) if x is None]
    if len(indices_to_generate) == 0:
        return None

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

            current_sequence = th.cat([current_sequence, th.tensor([[next_token]], device=model.device), next_sequence], dim=1)

        tube = model.idx_to_tube(th.tensor([[next_token]], device=model.device)).squeeze(0).tolist()
        yield json.dumps(tube)

    return None

@app.post("/structure")
async def get_structure(request: Request):
    data = await request.json()
    return StreamingResponse(structure_generation(data["data"], data["y"]))
