import torch as th
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel
from transformers.cache_utils import DynamicCache
from typing import Self


class SinkFormerConfig(LlamaConfig):
    model_type = "sinkformer"
    
    def __init__(
        self: Self,
        hidden_size: int = 512,
        intermediate_size: int = 1376,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 1024,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        num_special_tokens: int = 3,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(**kwargs)


class SinkFormer(LlamaModel):
    config_class = SinkFormerConfig

    def __init__(
        self: Self,
        config: SinkFormerConfig,
    ) -> None:
        self.sink_key_values = nn.Parameter(th.empty((config.num_hidden_layers, 1, 2, config.num_key_value_heads, config.hidden_size // config.num_attention_heads)))

        super().__init__(config)

    def forward(
        self: Self,
        inputs_embeds: th.Tensor,
        start_pos_indices: th.Tensor | None = None,
        attention_mask: th.Tensor | None = None,
        past_key_values: list[th.FloatTensor] | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        cache_position: th.LongTensor | None = None,
    ) -> th.Tensor:
        if start_pos_indices is None:
            position_ids = None
        else:
            position_ids = th.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            position_ids += start_pos_indices

        # set up initial sink key values
        if past_key_values is None or \
                (isinstance(past_key_values, list) and len(past_key_values[0]) == 0) or \
                (isinstance(past_key_values, DynamicCache) and past_key_values.seen_tokens == 0):
            past_key_values = [[self.sink_key_values[layer_idx]] for layer_idx in range(self.config.num_hidden_layers)] 
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        return super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
