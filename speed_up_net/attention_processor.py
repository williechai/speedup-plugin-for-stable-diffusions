# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")



class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def compute_attention_v1(attn, query, key, value, attention_mask):

    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)
    return hidden_states

def compute_attention_v2(attn, query, key, value, attention_mask):
    batch_size = query.shape[0]
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    return hidden_states


from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer

class SUNAttnProcessor(nn.Module):
    r"""
    Attention processor for SUN.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, defaults to 77):
            The context length of the original text-encoder features.
        add_pos (`bool`, defaults to False):
            Whether to use extra parameters for positive prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=77, param_scale=False, add_pos=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.add_pos = add_pos

        self.to_k_neg = LoRACompatibleLinear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_neg = LoRACompatibleLinear(cross_attention_dim or hidden_size, hidden_size, bias=False)    
        self.to_k_neg_alpha = nn.Parameter(torch.ones(hidden_size))
        self.to_k_neg_beta = nn.Parameter(torch.zeros(hidden_size))

        if self.add_pos:
            self.to_k_extra_pos = LoRACompatibleLinear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_extra_pos = LoRACompatibleLinear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_k_extra_pos_alpha = nn.Parameter(torch.ones(hidden_size))
            self.to_k_extra_pos_beta = nn.Parameter(torch.zeros(hidden_size))
        
        if param_scale:
            self.scale = nn.Parameter(torch.Tensor([scale]).float())
        else:
            self.scale = scale

    def set_cfg_scale(self, scale):
        g = (scale - 1) / scale
        self.scale = g

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, neg_hidden_states = encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:, :]
            if attn.norm_cross:
                assert False
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        neg_key = self.to_k_neg(neg_hidden_states)
        neg_value = self.to_v_neg(neg_hidden_states)

        if self.add_pos:
            pos_key = self.to_k_extra_pos(encoder_hidden_states)
            pos_value = self.to_v_extra_pos(encoder_hidden_states)
    
        if is_torch2_available():
            # print('v2, attn_mask:{}'.format(attention_mask))
            hidden_states = compute_attention_v2(attn, query, key, value, attention_mask)
            neg_hidden_states = compute_attention_v2(attn, query, neg_key, neg_value, None)
            if self.add_pos:
                pos_hidden_states = compute_attention_v2(attn, query, pos_key, pos_value, attention_mask)
        else:
            # print('v1, attn_mask:{}'.format(attention_mask))
            hidden_states = compute_attention_v1(attn, query, key, value, attention_mask)
            neg_hidden_states = compute_attention_v1(attn, query, neg_key, neg_value, None)
            if self.add_pos:
                pos_hidden_states = compute_attention_v1(attn, query, pos_key, pos_value, attention_mask)
        if torch.is_tensor(self.scale):
            shape = [-1] + [1] * (neg_hidden_states.ndim - 1)
            scale = self.scale.view(*shape)
        else:
            scale = self.scale

        
        origin_norm = torch.norm(hidden_states, p=2)
        neg_norm = torch.norm(neg_hidden_states, p=2)
        neg_hidden_states = origin_norm * (neg_hidden_states / neg_norm)
        neg_hidden_states = self.to_k_neg_alpha * neg_hidden_states + self.to_k_neg_beta

        if self.add_pos:
            pos_norm = torch.norm(pos_hidden_states, p=2)
            pos_hidden_states = origin_norm * (pos_hidden_states / pos_norm)
            pos_hidden_states = self.to_k_extra_pos_alpha * pos_hidden_states + self.to_k_extra_pos_beta

        hidden_states = hidden_states - neg_hidden_states + pos_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    

SUNAttnProcessor2_0 = SUNAttnProcessor
