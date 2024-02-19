import torch as th


def generate_binary_mapping(
    num_indices: int,
    num_bits: int,
) -> th.Tensor:
    return (th.arange(num_indices).unsqueeze(-1) >> th.arange(num_bits)) & 1  # (num_indices, num_bits)
