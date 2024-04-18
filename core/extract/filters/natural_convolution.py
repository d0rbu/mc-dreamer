import torch as th
from math import ceil
from torch.distributions import Normal


NATURAL_BLOCKS = [1, 2, 3, 12, 7, 13, 14, 15, 16, 31, 32, 37, 38, 48, 56, 73, 74, 78, 79, 80, 82, 97, 99, 100, 110, 129]


def natural_convolution_filter(
    tensor_blocks: th.ByteTensor,  # (512, 256, 512)
    mask: th.BoolTensor | None,  # unnecessary
    kernel_size: tuple[int, int, int] | int = (5, 5, 5),
    threshold: float = 0.9,
) -> th.BoolTensor:
    mask_values = th.tensor(NATURAL_BLOCKS)
    natural_blocks = th.isin(tensor_blocks, mask_values).float().unsqueeze(0).unsqueeze(0)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)

    # separability of gaussian kernel
    conv_kernel = gaussian_kernel(1.0, 3)
    conv_output = th.nn.functional.conv3d(natural_blocks, conv_kernel.view(1, 1, -1, 1, 1), padding="same")
    conv_output = th.nn.functional.conv3d(conv_output, conv_kernel.view(1, 1, 1, -1, 1), padding="same")
    conv_output = th.nn.functional.conv3d(conv_output, conv_kernel.view(1, 1, 1, 1, -1), padding="same")
    conv_output = conv_output.squeeze(0).squeeze(0)

    return conv_output < threshold  # only keep blocks that are not too natural

def gaussian_kernel(std: float, num_std: float = 3.) -> th.FloatTensor:
    radius = ceil(num_std * std)
    support = th.arange(-radius, radius + 1, dtype=th.float32)
    kernel = Normal(0, std).log_prob(support).exp()

    return kernel / kernel.sum()
