import xformers.ops as xops
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
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



def modify(modify_config):
    
    # 默认使用sdpa而不是eager
    if modify_config["type"] == "xformer":
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = xformer_forward
    elif modify_config["type"] == "keynorm":
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = keynorm_forward


if __name__ == "__main__":
    mask = xops.fmha.attn_bias.LowerTriangularFromBottomRightMask().materialize(shape=(2,4,8))
    print(mask)



"""
python ./models/model.py
"""
    