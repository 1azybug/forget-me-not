"""
https://github.com/huggingface/transformers/blob/ba56ed0869eb4bbeb1c04af7f62a04350150e8d4/src/transformers/models/llama/modeling_llama.py#L600
"""
import xformers.ops as xops
from transformers.modeling_outputs import BaseModelOutputWithPast
import transformers
import torch
import torch.nn.functional as F

def xformer_forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        cache_position,
        **kwargs,
    ):

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    cos, sin = self.rotary_emb(value_states, position_ids)  # value_states 只提供.dtype和.device.type
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

    
    key_states = torch.repeat_interleave(key_states, dim=2, repeats=self.num_key_value_groups)
    value_states = torch.repeat_interleave(value_states, dim=2, repeats=self.num_key_value_groups)

    # Input tensors must be in format [batch size, sequence length, number of heads, embeding size]
    assert query_states.size() == (bsz, q_len, self.num_heads, self.head_dim), "Input tensors must be in format [B, M, H, K], where B is the batch size, M the sequence length, H the number of heads, and K the embeding size per head"
    attn_output = xops.memory_efficient_attention(
        query_states, key_states, value_states,
        attn_bias=xops.fmha.attn_bias.LowerTriangularFromBottomRightMask()
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)


    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value



def keynorm_forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        cache_position,
        **kwargs,
    ):

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)\

    # 旋转位置编码不会改变模长
    key_states = F.normalize(key_states, p=2, dim=-1)

    cos, sin = self.rotary_emb(value_states, position_ids)  # value_states 只提供.dtype和.device.type
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
    
    key_states = torch.repeat_interleave(key_states, dim=2, repeats=self.num_key_value_groups)
    value_states = torch.repeat_interleave(value_states, dim=2, repeats=self.num_key_value_groups)

    # Input tensors must be in format [batch size, sequence length, number of heads, embeding size]
    assert query_states.size() == (bsz, q_len, self.num_heads, self.head_dim), "Input tensors must be in format [B, M, H, K], where B is the batch size, M the sequence length, H the number of heads, and K the embeding size per head"
    
    # scale set 1.0, because of KeyNorm( ||k|| ~= sqrt(d))
    attn_output = xops.memory_efficient_attention(
        query_states, key_states, value_states,
        scale=1.0,
        attn_bias=xops.fmha.attn_bias.LowerTriangularFromBottomRightMask()
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)


    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):

    # sin,cos: [1,kv_len,head_dim]
    # q:[batch_size,q_len,h_num,head_dim]
    # k:[batch_size,kv_len,h_num,head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # in cache setting, q_len < k_len
    q_len = q.size(1)
    q_embed = (q * cos[:,-q_len:]) + (rotate_half(q) * sin[:,-q_len:])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def cache_forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        cache_position,
        **kwargs,
    ):

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    # 旋转位置编码不会改变模长
    key_states = F.normalize(key_states, p=2, dim=-1)

    # cache the unratate kv
    key_states, value_states = past_key_value.update(key_states, value_states, layer_idx=self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)  # value_states 只提供.dtype和.device.type

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
    
    key_states = torch.repeat_interleave(key_states, dim=2, repeats=self.num_key_value_groups)
    value_states = torch.repeat_interleave(value_states, dim=2, repeats=self.num_key_value_groups)

    # Input tensors must be in format [batch size, sequence length, number of heads, embeding size]
    assert query_states.size() == (bsz, q_len, self.num_heads, self.head_dim), "Input tensors must be in format [B, M, H, K], where B is the batch size, M the sequence length, H the number of heads, and K the embeding size per head"
    # print(f"query_states shape:{query_states.shape}, key_states shape:{key_states.shape}")
    # scale set 1.0, because of KeyNorm( ||k|| ~= sqrt(d))
    attn_output = xops.memory_efficient_attention(
        query_states, key_states, value_states,
        scale=1.0,
        attn_bias=xops.fmha.attn_bias.LowerTriangularFromBottomRightMask()
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)


    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        cache_position = None,
    ):

    # [B,S,E]
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # input_len + cache_len
    tot_len = inputs_embeds.shape[1]+past_key_values.get_seq_length()
    # [1,input_len + cache_len]
    position_ids = torch.arange(0,tot_len,device=inputs_embeds.device).unsqueeze(0)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = None
    all_self_attns = None

    for decoder_layer in self.layers:

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = layer_outputs[0]

    hidden_states = self.norm(hidden_states)

    if not return_dict:
        return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def modify(modify_config):
    
    # 默认使用sdpa而不是eager
    if modify_config["type"] == "xformer":
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = xformer_forward
    elif modify_config["type"] == "keynorm":
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = keynorm_forward
    elif modify_config["type"] == 'cache':
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = cache_forward
        transformers.models.llama.modeling_llama.LlamaModel.forward = llama_forward


if __name__ == "__main__":
    mask = xops.fmha.attn_bias.LowerTriangularFromBottomRightMask().materialize(shape=(2,4,8))
    print(mask)



"""
python ./models/model.py
"""
    