import torch as th
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from typing import Self


class SinkFormerConfig(LlamaConfig):
    model_type = "sinkformer"
    
    def __init__(
        self: Self,
        vocab_size: int = 256,
        hidden_size: int = 512,
        intermediate_size: int = 1376,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        num_key_value_heads: int | None = None,
        max_position_embeddings: int = 1024,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        num_special_tokens: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = vocab_size + num_special_tokens
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_special_tokens = num_special_tokens


class SinkFormer(LlamaModel):
    config_class = SinkFormerConfig

    def __init__(
        self: Self,
        config: SinkFormerConfig,
    ) -> None:
        super().__init__(config)

        std = self.config.initializer_range
        # llama only initializes embeddings and linear layers for some reason
        self.sink_key_values = nn.Parameter(th.empty((config.num_hidden_layers, 2, 1, config.num_key_value_heads, 1, config.hidden_size // config.num_attention_heads)))
        self.sink_key_values.data.normal_(mean=0.0, std=std)

    def forward(
        self: Self,
        input_ids: th.LongTensor,
        start_pos_indices: th.Tensor | None = None,
        position_ids: th.LongTensor | None = None,
        attention_mask: th.Tensor | None = None,
        past_key_values: list[th.FloatTensor] | None = None,
        inputs_embeds: th.FloatTensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        cache_position: th.LongTensor | None = None,
        **kwargs,
    ) -> th.Tensor:
        if position_ids is None and start_pos_indices is not None:
            position_ids = th.arange(input_ids.shape[1], device=input_ids.device)
            position_ids += start_pos_indices

        # set up initial sink key values
        if past_key_values is None or \
                (isinstance(past_key_values, list) and len(past_key_values[0]) == 0) or \
                (isinstance(past_key_values, DynamicCache) and past_key_values.seen_tokens == 0):
            batch_size = input_ids.shape[0]
            past_key_values = [[*self.sink_key_values[layer_idx].expand(-1, batch_size, -1, -1, -1)] for layer_idx in range(self.config.num_hidden_layers)]
        
        if "use_cache" in kwargs:
            del kwargs["use_cache"]

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )


class CausalSinkFormer(LlamaForCausalLM):
    def __init__(
        self: Self,
        config: SinkFormerConfig
    ) -> None:
        super(LlamaForCausalLM, self).__init__(config)
        self.model = SinkFormer(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
