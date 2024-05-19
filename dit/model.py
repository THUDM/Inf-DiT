from omegaconf import OmegaConf
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from sat.model.base_model import BaseModel
from sat.model.transformer import CrossAttention
from sat.model.mixins import BaseMixin
from sat.helpers import print_rank0
from sat.transformer_defaults import standard_attention

from sat.ops.layernorm import LayerNorm

from dit.embeddings import TimeEmbedding, ConditionEmbedding, ImagePatchEmbeddingMixin, BasicPositionEmbeddingMixin
from dit.embeddings import DDPMTimeEmbedding, RotaryPositionEmbedding

from dit.utils import instantiate_from_config, append_dims
from sat.mpu.utils import split_tensor_along_last_dim
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from sat.mpu.utils import unscaled_init_method


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def unpatchify(x, c, p, rope_position_ids=None, hw=None):
    """
    x: (N, T, patch_size**2 * C)
    imgs: (N, H, W, C)
    """
    if False:
        # do pix2struct unpatchify
        L = x.shape[1]
        x = x.reshape(shape=(x.shape[0], L, p, p, c))
        x = torch.einsum('nlpqc->ncplq', x)
        imgs = x.reshape(shape=(x.shape[0], c, p, L * p))
    else:
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
    
    return imgs


class FinalLayerMixin(BaseMixin):
    def __init__(self, hidden_size, patch_size, num_patches, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    
    def final_forward(self, logits, **kwargs):
        x, emb = logits, kwargs['emb']
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return unpatchify(x, c=self.out_channels, p=self.patch_size, rope_position_ids=kwargs.get('rope_position_ids', None), hw=kwargs.get('hw', None))

    def reinit(self, parent_model=None):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

class SimpleFinalLayerMixin(BaseMixin):
    def __init__(self, hidden_size, patch_size, num_patches, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def final_forward(self, logits, **kwargs):
        x = self.linear(logits)
        return unpatchify(x, c=self.out_channels, p=self.patch_size, rope_position_ids=kwargs.get('rope_position_ids', None))

    def reinit(self, parent_model=None):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

class AdaLNMixin(BaseMixin):
    def __init__(self, hidden_size, num_layers, image_size, pos_embed_config, is_decoder=False,
                 nogate=False, cross_adaln=False, use_block_attention=False,
                 num_patches=None, qk_ln=False, num_head=None, random_position=False, block_size=None,
                 re_position=False, cross_lr=False, args=None, params_dtype=torch.float32):
        super().__init__()
        self.nogate = nogate
        self.cross_adaln = cross_adaln
        self.image_size = image_size
        self.num_patches = num_patches
        self.block_size = block_size
        self.sr_scale = args.sr_scale
        self.patch_size = args.patch_size
        if nogate:
            out_times = 4
        else:
            out_times = 6
        if cross_adaln:
            out_times = (out_times // 2) * 3

        self.qk_ln = qk_ln
        hidden_size_head = hidden_size // num_head
        if qk_ln:
            print("--------use qk_ln--------")
            self.q_layer_norm = nn.ModuleList([
                nn.LayerNorm(hidden_size_head, eps=1e-6)
                for _ in range(num_layers)
            ])
            self.k_layer_norm = nn.ModuleList([
                nn.LayerNorm(hidden_size_head, eps=1e-6)
                for _ in range(num_layers)
            ])

        self.adaLN_modulations = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, out_times * hidden_size)
            ) for _ in range(num_layers)
            ])
        self.is_decoder = is_decoder
        self.use_block_attention = use_block_attention
        if random_position:
            self.rope = RotaryPositionEmbedding(pix2struct=True, num_patches=image_size * 8, hidden_size=hidden_size, hidden_size_head=hidden_size//num_head)
        else:
            self.rope = RotaryPositionEmbedding(pix2struct=True, num_patches=image_size * 2, hidden_size=hidden_size,
                                                hidden_size_head=hidden_size // num_head)
        self.re_position = re_position

        if re_position:
            self.re_pos_embed = nn.Parameter(torch.zeros(num_layers, 4, num_head, hidden_size_head))

        self.cross_lr = cross_lr
        if cross_lr:
            in_channels = 3 # latent is 4
            self.lr_patch_size = 2
            bias = True
            self.proj_lr = nn.Conv2d(in_channels, hidden_size, kernel_size=self.lr_patch_size, stride=self.lr_patch_size, bias=bias)
            if hasattr(args, 'fp16') and args.fp16:
                params_dtype = torch.half
            elif hasattr(args, 'bf16') and args.bf16:
                params_dtype = torch.bfloat16
            else:
                params_dtype = torch.float32
            self.cross_attention = CrossAttention(
                hidden_size,
                args.num_attention_heads,
                0,
                0,
                unscaled_init_method(0.02),
                0,
                hidden_size_per_attention_head=None,
                output_layer_init_method=unscaled_init_method(0.02),
                cross_attn_hidden_size=None,
                bias=True,
                hooks=None,
                transformer_pointer=self,
                params_dtype=params_dtype
            )
            self.post_cross_attention_layernorm = LayerNorm(hidden_size, eps=1e-6)
            head = args.num_attention_heads
            hidden_size_per_attention_head = hidden_size // head
            self.lr_query_position_embedding = nn.Parameter(torch.zeros(1, self.block_size ** 2, head, hidden_size_per_attention_head))
            self.lr_block_size = self.block_size * self.patch_size // self.sr_scale // self.lr_patch_size
            self.lr_key_position_embedding = nn.Parameter(torch.zeros(1, (self.lr_block_size * 3) ** 2, head, hidden_size_per_attention_head))
            torch.nn.init.normal_(self.lr_query_position_embedding, std=0.02)
            torch.nn.init.normal_(self.lr_key_position_embedding, std=0.02)
            if self.qk_ln:
                self.cross_q_ln = nn.LayerNorm(hidden_size_head, eps=1e-6)
                self.cross_k_ln = nn.LayerNorm(hidden_size_head, eps=1e-6)

    def position_embedding_forward(self, position_ids, **kwargs):
        return None

    def attention_forward(self, hidden_states, mask, rope_position_ids, inference=0, direction="lt", mems=None, do_concat=True, **kw_args):
        origin = self
        self = self.transformer.layers[kw_args['layer_id']].attention

        mixed_raw_layer = self.query_key_value(hidden_states)
        (query_layer,
         key_layer,
         value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        dropout_fn = self.attention_dropout if self.training else None

        # [b, h, n*n, d/h]
        h, w = kw_args['hw']
        if origin.use_block_attention:
            mask = None
            block_size = origin.block_size
            batch_size = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            in_x = h // block_size
            in_y = w // block_size
            query_layer = query_layer.view(batch_size, seq_len, -1, self.hidden_size_per_attention_head)
            key_layer = key_layer.view(batch_size, seq_len, -1, self.hidden_size_per_attention_head)
            value_layer = value_layer.view(batch_size, seq_len, -1, self.hidden_size_per_attention_head)
            if origin.qk_ln:
                query_layernorm = origin.q_layer_norm[kw_args['layer_id']]
                key_layernorm = origin.q_layer_norm[kw_args['layer_id']]
                query_layer = query_layernorm(query_layer)
                key_layer = key_layernorm(key_layer)

            if origin.rope:
                query_layer = origin.rope(query_layer, rope_position_ids=rope_position_ids)
                key_layer = origin.rope(key_layer, rope_position_ids=rope_position_ids)

            def transform(x):
                x = rearrange(x, 'b (n m) h d -> b n m h d', n=h, m=w)
                x = rearrange(x, 'b (x l) (y w) h d -> b x y (l w) h d', l=block_size, w=block_size)
                return x

            def inverse_transform(x):
                x = rearrange(x, '(b x y) (l w) h d -> b (x l y w) (h d)', l=block_size, w=block_size, x=in_x, y=in_y)
                return x

            def get_cat_layer_left_top(x):
                cat_x = torch.cat([x[:, 0:1], x], dim=1)
                cat_x = torch.cat([cat_x[:, :, 0:1], cat_x], dim=2)
                x = torch.cat([x, cat_x[:, :x.shape[1], :x.shape[2]],
                               cat_x[:, 1:, :x.shape[2]], cat_x[:, :x.shape[1], 1:]], dim=3)
                #self lef-top left top
                return x

            def get_cat_layer_right_bottom(x):
                cat_x = torch.cat([x, x[:, -1:]], dim=1)
                cat_x = torch.cat([cat_x, cat_x[:, :, -1:]], dim=2)
                x = torch.cat([x, cat_x[:, 1:, 1:], cat_x[:, 1:, :x.shape[2]],
                               cat_x[:, :x.shape[1], 1:]], dim=3)
                return x

            def get_cat_layer_left_top_withcache(x, mems, index):
                if mems[2] is not None: # top
                    cat_x = torch.cat([mems[2][kw_args['layer_id']]['mem_kv'][index], x], dim=1)
                else:
                    cat_x = torch.cat([x[:, 0:1], x], dim=1)
                if mems[0] is not None: #left
                    if mems[1] is not None: # left-top
                        left_cat = torch.cat([mems[1][kw_args['layer_id']]['mem_kv'][index][:, :, -1:], mems[0][kw_args['layer_id']]['mem_kv'][index+2]], dim=1)
                    else:
                        left_cat = torch.cat([mems[0][kw_args['layer_id']]['mem_kv'][index+2][:, 0:1], mems[0][kw_args['layer_id']]['mem_kv'][index+2]], dim=1)
                    cat_x = torch.cat([left_cat, cat_x], dim=2)
                else:
                    cat_x = torch.cat([cat_x[:, :, 0:1], cat_x], dim=2)
                x = torch.cat([x, cat_x[:, :x.shape[1], :x.shape[2]],
                               cat_x[:, 1:, :x.shape[2]], cat_x[:, :x.shape[1], 1:]], dim=3)
                return x

            query_layer = transform(query_layer)
            key_layer = transform(key_layer)
            value_layer = transform(value_layer)

            if not do_concat:
                pass
            elif inference == 1:
                kw_args['output_this_layer']['mem_kv'] = [key_layer, value_layer]
                k_stack = [key_layer]
                v_stack = [value_layer]
                for mem in mems:
                    k_stack.append(mem[kw_args['layer_id']]['mem_kv'][0])
                    v_stack.append(mem[kw_args['layer_id']]['mem_kv'][1])
                key_layer = torch.cat(k_stack, dim=3)
                value_layer = torch.cat(v_stack, dim=3)
            elif inference == 2:
                # mems:left, left-top, top
                kw_args['output_this_layer']['mem_kv'] = [key_layer[:, -1:].clone(), value_layer[:, -1:].clone(), key_layer[:, :, -1:].clone(), value_layer[:, :, -1:].clone()]
                key_layer = get_cat_layer_left_top_withcache(key_layer, mems, 0)
                value_layer = get_cat_layer_left_top_withcache(value_layer, mems, 1)
            elif direction == "lt":
                key_layer = get_cat_layer_left_top(key_layer)
                value_layer = get_cat_layer_left_top(value_layer)
            elif direction == "rb":
                key_layer = get_cat_layer_right_bottom(key_layer)
                value_layer = get_cat_layer_right_bottom(value_layer)
            else:
                raise NotImplementedError

            query_layer = query_layer.view(-1, *query_layer.shape[3:])
            key_layer = key_layer.view(-1, *key_layer.shape[3:])
            value_layer = value_layer.view(-1, *value_layer.shape[3:])

            if origin.re_position and do_concat:
                re_pos_embed = origin.re_pos_embed[kw_args['layer_id']].repeat_interleave(key_layer.shape[1] // origin.re_pos_embed.shape[1], dim=0).unsqueeze(0)
                key_layer = key_layer + re_pos_embed

            context_layer = flash_attn_func(query_layer, key_layer, value_layer)

            context_layer = inverse_transform(context_layer)
        else:
            query_layer = self._transpose_for_scores(query_layer)
            key_layer = self._transpose_for_scores(key_layer)
            value_layer = self._transpose_for_scores(value_layer)
            context_layer = standard_attention(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)

        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output

    def process_lr(self, lr_imgs):
        lr_imgs = self.proj_lr(lr_imgs)
        lr_hidden_size = lr_imgs.shape[1]

        unFold = torch.nn.Unfold(kernel_size=3 * self.lr_block_size, stride=self.lr_block_size,
                                 padding=self.lr_block_size)
        lr_imgs = unFold(lr_imgs)

        lr_imgs = lr_imgs.view(lr_imgs.shape[0], lr_hidden_size, self.lr_block_size * 3, self.lr_block_size * 3, -1)
        lr_imgs = lr_imgs.permute(0, 4, 2, 3, 1).contiguous()
        lr_imgs = lr_imgs.view(lr_imgs.shape[0] * lr_imgs.shape[1], -1, lr_imgs.shape[-1])
        return lr_imgs

    def cross_attention_forward(self, hidden_states, lr_imgs, **kw_args):
        h, w = kw_args['hw']
        block_size = self.block_size
        in_x = h // block_size
        in_y = w // block_size
        def transform(x):
            x = rearrange(x, 'b (n m) h d -> b n m h d', n=h, m=w)
            x = rearrange(x, 'b (x l) (y w) h d -> b x y (l w) h d', l=block_size, w=block_size)
            return x

        def inverse_transform(x):
            x = rearrange(x, '(b x y) (l w) h d -> b (x l y w) (h d)', l=block_size, w=block_size, x=in_x, y=in_y)
            return x

        origin = self
        self = self.cross_attention
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        query_layer = self.query(hidden_states)
        mixed_x_layer = self.key_value(lr_imgs)
        key_layer, value_layer = split_tensor_along_last_dim(mixed_x_layer, 2)
        query_layer = query_layer.view(batch_size, seq_len, -1, self.hidden_size_per_attention_head)

        key_layer = key_layer.view(key_layer.shape[0], lr_imgs.shape[1], -1, self.hidden_size_per_attention_head)
        value_layer = value_layer.view(key_layer.shape[0], lr_imgs.shape[1], -1, self.hidden_size_per_attention_head)


        query_layer = transform(query_layer)
        query_layer = query_layer.view(-1, *query_layer.shape[3:])

        if origin.qk_ln:
            query_layer = origin.cross_q_ln(query_layer)
            key_layer = origin.cross_k_ln(key_layer)

        query_layer = query_layer + origin.lr_query_position_embedding
        key_layer = key_layer + origin.lr_key_position_embedding

        context_layer = flash_attn_func(query_layer, key_layer, value_layer)
        context_layer = inverse_transform(context_layer)
        # Output. [b, s, h]
        output = self.dense(context_layer)
        if self.training:
            output = self.output_dropout(output)
        return output

    def layer_forward(self, hidden_states, mask, do_concat=True, *args, **kwargs):
        layer_id = kwargs['layer_id']
        layer = self.transformer.layers[kwargs['layer_id']]
        adaLN_modulation = self.adaLN_modulations[kwargs['layer_id']]
        if self.nogate and self.cross_adaln:
            shift_msa, scale_msa, shift_cross, scale_cross, shift_mlp, scale_mlp = adaLN_modulation(kwargs['emb']).chunk(6, dim=1)
            gate_msa = gate_cross = gate_mlp = 1
        elif self.nogate and not self.cross_adaln:
            shift_msa, scale_msa, shift_mlp, scale_mlp = adaLN_modulation(kwargs['emb']).chunk(4, dim=1)
            gate_msa = gate_mlp = 1
        elif not self.nogate and self.cross_adaln:
            shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = adaLN_modulation(kwargs['emb']).chunk(9, dim=1)
            gate_msa, gate_cross, gate_mlp = gate_msa.unsqueeze(1), gate_cross.unsqueeze(1), gate_mlp.unsqueeze(1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN_modulation(kwargs['emb']).chunk(6, dim=1)
            gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)

        # self attention
        attention_input = layer.input_layernorm(hidden_states)
        attention_input = modulate(attention_input, shift_msa, scale_msa)

        concat = kwargs.get('concat', None)
        if concat is not None and concat[0] != concat[-1]:
            input1, input2 = attention_input.chunk(2)
            rope_position_ids1, rope_position_ids2 = kwargs['rope_position_ids'].chunk(2)
            kwargs['rope_position_ids'] = rope_position_ids1
            output1 = layer.attention(input1, mask, do_concat=concat[0], **kwargs)
            kwargs['rope_position_ids'] = rope_position_ids2
            output2 = layer.attention(input2, mask, do_concat=concat[-1], **kwargs)
            attention_output = torch.cat([output1, output2], dim=0)
        else:
            attention_output = layer.attention(attention_input, mask, do_concat=do_concat, **kwargs)

        if self.transformer.layernorm_order == 'sandwich':
            attention_output = layer.third_layernorm(attention_output)
        hidden_states = hidden_states + gate_msa * attention_output
        if layer_id == 0 and self.cross_lr:
            cross_attention_input = layer.post_attention_layernorm(hidden_states)

            # do cross attention here
            cross_attention_output = self.cross_attention_forward(cross_attention_input, **kwargs)

            hidden_states = hidden_states + cross_attention_output
            mlp_input = self.post_cross_attention_layernorm(hidden_states)
        else:
            mlp_input = layer.post_attention_layernorm(hidden_states)

        mlp_input = modulate(mlp_input, shift_mlp, scale_mlp)
        mlp_output = layer.mlp(mlp_input, **kwargs)
        if self.transformer.layernorm_order == 'sandwich':
            mlp_output = layer.fourth_layernorm(mlp_output)
        hidden_states = hidden_states + gate_mlp * mlp_output
        return hidden_states

    def reinit(self, parent_model=None):
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)
class DiffusionEngine(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        self.image_size = args.image_size
        self.patch_size = args.patch_size
        self.num_patches = (args.image_size // args.patch_size)**2
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.scale_factor = args.scale_factor
        self.input_time = args.input_time
        self.is_decoder = args.is_decoder
        self.no_crossmask = args.no_crossmask
        self.stop_grad_patch_embed = args.stop_grad_patch_embed
        self.sr_scale = args.sr_scale
        self.random_position = args.random_position
        self.random_direction = args.random_direction
        self.image_block_size = args.image_block_size
        if self.random_direction:
            print("--------use random direction--------")
        if self.random_position:
            print("--------use random position--------")

        self.use_block_attention = True
        
        if 'activation_func' not in kwargs:
            approx_gelu = nn.GELU(approximate='tanh')
            kwargs['activation_func'] = approx_gelu
        args.parallel_output = parallel_output
        super().__init__(args, transformer=transformer, layernorm=partial(LayerNorm, elementwise_affine=False, eps=1e-6),  **kwargs)
        
        configs = OmegaConf.load(args.config_path)
        module_configs = configs.pop('modules', None)
        modeling_configs = configs.pop('modeling', None)


        self._build_modules(args, module_configs)
        self._build_modeling(args, modeling_configs)

        self.collect_attention = None
    def _build_modules(self, args, module_configs):

        pos_embed_config = module_configs.pop('position_embedding_config')
        if self.input_time == "adaln":
            self.add_mixin('adaln_layer', AdaLNMixin(args.hidden_size, args.num_layers, is_decoder=args.is_decoder,
                                                     image_size=args.image_size//args.patch_size,
                                                     nogate=args.nogate, cross_adaln=args.cross_adaln,
                                                     use_block_attention=self.use_block_attention, pos_embed_config=pos_embed_config,
                                                     num_patches=self.num_patches, num_head=args.num_attention_heads,
                                                     qk_ln=args.qk_ln, random_position=args.random_position,
                                                     block_size=args.image_block_size//args.patch_size,
                                                     re_position=args.re_position,
                                                     cross_lr=args.cross_lr, args=args), reinit=True)
            self.add_mixin('final_layer', FinalLayerMixin(args.hidden_size, args.patch_size, self.num_patches, args.out_channels), reinit=True)
        elif self.input_time in ["concat", "add"]:
            self.add_mixin('final_layer', SimpleFinalLayerMixin(args.hidden_size, args.patch_size, self.num_patches, args.out_channels), reinit=True)
        else:
            raise NotImplementedError

        append_emb = self.input_time == "concat"
        add_emb = self.input_time == "add"

        self.cross_lr = args.cross_lr
        self.add_mixin('patch_embed', ImagePatchEmbeddingMixin(args.in_channels, args.hidden_size, args.patch_size, append_emb=append_emb, add_emb=add_emb, reg_token_num=args.reg_token_num), reinit=True)


        scale_factor = module_configs.pop('scale_factor', 1)
        self.scale_factor = scale_factor
        first_stage_config = module_configs.pop('first_stage_config', None)
        if first_stage_config:
            self.first_stage_model = instantiate_from_config(first_stage_config)
            self._init_first_stage()
        else:
            self.first_stage_model = None

        if args.ddpm_time_emb:
            self.time_embed = DDPMTimeEmbedding(args.hidden_size)
        else:
            self.time_embed = TimeEmbedding(args.hidden_size)
        self.cond_embed = ConditionEmbedding(args.hidden_size, args.label_dim, args.augment_dim, args.vector_dim, args.label_dropout)
        
        if args.is_decoder:
            text_encoder_config = module_configs.pop('text_encoder_config')
            self.text_encoder = instantiate_from_config(text_encoder_config)
        else:
            self.text_encoder = None

        if args.image_condition:
            image_encoder_config = module_configs.pop('image_encoder_config')
            self.image_encoder = instantiate_from_config(image_encoder_config)
        else:
            self.image_encoder = None

    def _build_modeling(self, args, modeling_configs):
        precond_config = modeling_configs.pop('precond_config')
        self.precond = instantiate_from_config(precond_config)
        
        loss_config = modeling_configs.pop('loss_config')
        self.loss_func = instantiate_from_config(loss_config)
        sampler_config = modeling_configs.pop('sampler_config')
        self.sampler = instantiate_from_config(sampler_config)
    
    def disable_untrainable_params(self):
        disable_prefixs = ["text_encoder", "first_stage_model", "image_encoder"]
        total_trainable = 0
        for n, p in self.named_parameters():
            flag = False
            for prefix in disable_prefixs:
                if n.startswith(prefix):
                    flag = True
                    break
            if flag:
                p.requires_grad_(False)
            else:
                total_trainable += p.numel()
        if self.stop_grad_patch_embed:
            self.mixins.patch_embed.proj.weight.requires_grad_(False)
            self.mixins.patch_embed.proj.bias.requires_grad_(False)
            total_trainable -= self.mixins.patch_embed.proj.weight.numel()
            total_trainable -= self.mixins.patch_embed.proj.bias.numel()

        print_rank0("***** Total trainable parameters: "+str(total_trainable)+" *****")
                
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('Diffusion', 'Diffusion Configurations')
        group.add_argument('--image-size', type=int, default=64)
        group.add_argument('--patch-size', type=int, default=4)
        group.add_argument('--in-channels', type=int, default=3)
        group.add_argument('--out-channels', type=int, default=3)
        group.add_argument('--scale-factor', type=float, default=1)
        group.add_argument('--cross-attn-hidden-size', type=int, default=640)
        group.add_argument('--augment-dim', type=int, default=0)
        group.add_argument('--label-dim', type=int, default=0)
        group.add_argument('--label-dropout', type=float, default=0)
        group.add_argument('--is-decoder', action='store_true')
        group.add_argument('--input-time', type=str, choices=["adaln", "concat", "concatv2", "add"], default="adaln")
        group.add_argument('--ddpm-time-emb', action='store_true')
        group.add_argument('--reg-token-num', type=int, default=0)
        group.add_argument('--nogate', action='store_true')
        group.add_argument('--cross-adaln', action='store_true')
        group.add_argument('--no-crossmask', action='store_true')
        group.add_argument('--stop-grad-patch-embed', action='store_true')
        group.add_argument('--text-dropout', type=float, default=0)
        group.add_argument('--config-path', type=str, default=None)
        group.add_argument('--sr-scale', type=int, default=4)
        group.add_argument('--qk-ln', action='store_true')
        group.add_argument('--random-position', action='store_true')
        group.add_argument('--random-direction', action='store_true')
        group.add_argument('--image-block-size', type=int, default=128)
        group.add_argument('--vector-dim', type=int, default=0)
        group.add_argument('--image-condition', action='store_true')
        group.add_argument('--lr-dropout', default=0, type=float)
        group.add_argument('--re-position', action='store_true')
        group.add_argument('--cross-lr', action='store_true')
        return parser
    
    @classmethod
    def get_data_config(cls, args):
        configs = OmegaConf.load(args.config_path)
        data_config = configs.get('data', {})
        return data_config
        
    def model_forward(self, *args, **kwargs):
        sigmas = kwargs["sigmas"]
        emb = self.time_embed(sigmas)
        emb = self.cond_embed(emb, **kwargs)
        emb = F.silu(emb)
        kwargs['emb'] = emb
        
        kwargs['input_ids'] = kwargs['position_ids'] = kwargs['attention_mask'] = torch.ones((1,1)).to(sigmas.dtype)
        return super().forward(*args, **kwargs)
    
    def precond_forward(self, inference, rope_position_ids, concat_lr_imgs, lr_imgs=None, ar=False,
                        ar2=False, sample_step=None, block_batch=1, *args, **kwargs):
        images, sigmas = kwargs["images"], kwargs["sigmas"]

        images = torch.cat((images, concat_lr_imgs), dim=1)

        # TODO: reduce LR image memory cost
        # lr_imgs = lr_imgs[:, :, :128, :128]
        if self.mixins['adaln_layer'].cross_lr:
            lr_imgs = self.mixins['adaln_layer'].process_lr(lr_imgs)


        h, w = images.shape[2:4]
        c_skip, c_out, c_in, c_noise = map(lambda t: t.to(images.dtype), self.precond(append_dims(sigmas, images.ndim)))

        images *= self.scale_factor

        if inference and ar: # block_batch=1
            assert block_batch == 1
            block_size = self.image_block_size
            vit_block_size = block_size // self.patch_size
            block_h, block_w = h // block_size, w // block_size
            rope_position_ids = rope_position_ids.view(-1, h // self.patch_size, w // self.patch_size, 2)
            samples = []
            cached = [None] * block_w
            images = images
            if self.random_direction and sample_step is not None and sample_step % 2 == 1:
                range_i = range(block_h - 1, -1, -1)
                range_j = range(block_w - 1, -1, -1)
            else:
                range_i = range(block_h)
                range_j = range(block_w)
            for i in range_i:
                previous = None
                sample_row = []
                for j in range_j:
                    tmp_images = images[:, :, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    tmp_position_ids = rope_position_ids[:, i*vit_block_size:(i+1)*vit_block_size, j*vit_block_size:(j+1)*vit_block_size].contiguous().view(-1, vit_block_size * vit_block_size, 2)
                    kwargs["images"] = tmp_images * c_in
                    kwargs["sigmas"] = c_noise.reshape(-1)
                    kwargs["rope_position_ids"] = tmp_position_ids
                    mems = []
                    if cached[j] is not None:
                        mems.append(cached[j])
                    if j != 0:
                        if cached[j-1] is not None:
                            mems.append(cached[j-1])
                        mems.append(previous)
                    lr_id = i * block_w + j
                    output, *output_per_layers = self.model_forward(*args, hw=[vit_block_size, vit_block_size], mems=mems, inference=1, lr_imgs=lr_imgs[lr_id:lr_id+1], **kwargs)
                    output = output * c_out + tmp_images[:, :self.out_channels] * c_skip
                    if j != 0:
                        cached[j-1] = previous
                    if j == block_w - 1:
                        cached[j] = output_per_layers
                    else:
                        previous = output_per_layers
                    sample_row.append(output)

                sample_row = torch.cat(sample_row, dim=3)
                samples.append(sample_row)
            samples = torch.cat(samples, dim=2)
            return 1. / self.scale_factor * samples
        elif inference and ar2:  # block_batch>1
            block_size = self.image_block_size  # hard code
            vit_block_size = block_size // self.patch_size
            block_h, block_w = h // block_size, w // block_size
            rope_position_ids = rope_position_ids.view(-1, h // self.patch_size, w // self.patch_size, 2)
            samples = []
            cached = [None] * block_w
            images = images

            block_batch = 16
            block_bsize = block_size * block_batch
            vit_block_bsize = vit_block_size * block_batch
            assert block_h % block_batch == 0
            assert block_w % block_batch == 0
            block_batch_h = block_h // block_batch
            block_batch_w = block_w // block_batch
            range_i = range(block_batch_h)
            range_j = range(block_batch_w)
            for i in range_i:
                previous = None
                sample_row = []
                for j in range_j:
                    tmp_images = images[:, :, i * block_bsize:(i + 1) * block_bsize,
                                 j * block_bsize:(j + 1) * block_bsize]
                    tmp_position_ids = rope_position_ids[:, i * vit_block_bsize:(i + 1) * vit_block_bsize,
                                       j * vit_block_bsize:(j + 1) * vit_block_bsize].contiguous().view(-1, vit_block_bsize * vit_block_bsize, 2)
                    kwargs["images"] = tmp_images * c_in
                    kwargs["sigmas"] = c_noise.reshape(-1)
                    kwargs["rope_position_ids"] = tmp_position_ids
                    mems = [None, None, None]
                    def get_top_concat(x):
                        return [{"mem_kv": y['mem_kv'][:2]} for y in x]
                    if j != 0:
                        mems[0] = previous
                        if cached[j - 1] is not None:
                            mems[1] = cached[j - 1]
                    if cached[j] is not None:
                        mems[2] = cached[j]
                    output, *output_per_layers = self.model_forward(*args, hw=[vit_block_bsize,
                                                                               vit_block_bsize],
                                                                    mems=mems, inference=2,  **kwargs)
                    output = output * c_out + tmp_images[:, :self.out_channels] * c_skip
                    if previous is not None:
                        previous = get_top_concat(previous)
                    if j != 0:
                        cached[j - 1] = previous
                    if j == block_w - 1:
                        cached[j] = get_top_concat(output_per_layers)
                    else:
                        previous = output_per_layers
                    sample_row.append(output)

                sample_row = torch.cat(sample_row, dim=3)
                samples.append(sample_row)
            samples = torch.cat(samples, dim=2)
            return 1. / self.scale_factor * samples
        else:
            kwargs["images"] = images * c_in
            kwargs["sigmas"] = c_noise.reshape(-1)
            direction = "lt"
            if self.random_direction and torch.rand(1) > 0.5 and not inference:
                direction = "rb"
            if self.random_direction and sample_step is not None and sample_step % 2 == 1:
                direction = "rb"
            # switch direction
            # if direction == "rb":
            #     direction = "lt"
            # else:
            #     direction = "rb"
            kwargs["direction"] = direction
            output, *output_per_layers = self.model_forward(*args, hw=[h//self.patch_size, w//self.patch_size], rope_position_ids=rope_position_ids, lr_imgs=lr_imgs, **kwargs)
            output = output * c_out + images[:, :self.out_channels] * c_skip

            #calc attention score
            if self.collect_attention is not None:
                self.collect_attention.append(output_per_layers)

            # output = output.to(in_type)
            return 1. / self.scale_factor * output

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        out = self.first_stage_model.decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        z = self.first_stage_model.encode(x)
        z = self.scale_factor * z
        return z

    def _init_first_stage(self):
        self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        if self.is_decoder:
            sigmas = kwargs["sigmas"]
            self.text_encoder = self.text_encoder.to(self.text_encoder.encoder_dtype)
            with torch.no_grad():
                context, context_mask = self.text_encoder(**kwargs['text_inputs'])
            if self.no_crossmask:
                context_mask = torch.ones_like(context_mask)
            kwargs['encoder_outputs'] = context.to(sigmas.dtype)
            kwargs['cross_attention_mask'] = context_mask.to(sigmas.dtype)

        device = kwargs["images"].device
        # add position_ids
        position_ids = torch.zeros(self.num_patches, 2, device=device)
        position_ids[:, 0] = torch.arange(self.num_patches) // (self.image_size // self.patch_size)
        position_ids[:, 1] = torch.arange(self.num_patches) % (self.image_size // self.patch_size)
        if self.random_position:
            # change to 15 for long context
            position_ids[:, 0] += torch.randint(0, 7 * (self.image_size // self.patch_size) + 1, (1,), device=device)
            position_ids[:, 1] += torch.randint(0, 7 * (self.image_size // self.patch_size) + 1, (1,), device=device)
        position_ids = torch.repeat_interleave(position_ids.unsqueeze(0), kwargs["images"].shape[0], dim=0).long()
        position_ids[position_ids==-1] = 0

        return self.precond_forward(rope_position_ids=position_ids, inference=False, *args, **kwargs)
    

    def sample(
        self,
        shape,
        rope_position_ids=None,
        num_steps=None,
        images=None,
        lr_imgs=None,
        init_noise=True,
        dtype=torch.float32,
        device=torch.device('cuda'),
        return_attention_map=False,
        image_2=None,
        do_concat=True,
        ar=False,
        ar2=False,
        block_batch=1,
    ):

        if images is None:
            images = torch.randn(*shape).to(dtype).to(device)

        cond = {}
        uncond = {}

        if image_2 is not None:
            cond["image2"] = image_2

        if self.image_encoder:
            if image_2 is not None:
                print("has image2!!!")
                image_embedding = self.image_encoder(image_2)
            else:
                image_embedding = self.image_encoder(lr_imgs, add_text=True)
            cond["vector"] = image_embedding
            uncond["vector"] = image_embedding

        cond["lr_imgs"] = lr_imgs
        uncond["lr_imgs"] = lr_imgs

        #for cat cfg
        cond["concat"] = torch.ones(1, dtype=torch.bool, device=images.device)
        uncond["concat"] = torch.zeros(1, dtype=torch.bool, device=images.device)

        cond["concat_lr_imgs"] = images
        uncond["concat_lr_imgs"] = images

        h, w = images.shape[2:4]
        num_patches = h * w // (self.patch_size ** 2)
        if rope_position_ids is None:
            position_ids = torch.zeros(num_patches, 2, device=images.device)
            position_ids[:, 0] = torch.arange(num_patches, device=images.device) // (w // self.patch_size)
            position_ids[:, 1] = torch.arange(num_patches, device=images.device) % (w // self.patch_size)

            position_ids = position_ids % (8 * self.image_size // self.patch_size)

            position_ids = torch.repeat_interleave(position_ids.unsqueeze(0), images.shape[0], dim=0).long()
            position_ids[position_ids == -1] = 0
            rope_position_ids = position_ids

        self.transformer.output_hidden_states = True

        denoiser = lambda images, sigmas, rope_position_ids, cond, sample_step: self.precond_forward(images=images,
            sigmas=sigmas, rope_position_ids=rope_position_ids, inference=True, sample_step=sample_step, do_concat=do_concat,
                                                                                                     ar=ar, ar2=ar2, block_batch=block_batch, **cond)

        if return_attention_map:
            self.collect_attention = []
        samples = self.sampler(denoiser=denoiser, x=None, cond=cond, uc=uncond, num_steps=num_steps, rope_position_ids=rope_position_ids, init_noise=init_noise)

        if self.first_stage_model:
            samples = self.decode_first_stage(samples)

        self.transformer.output_hidden_states = False
        if return_attention_map:
            attention_maps = self.collect_attention
            self.collect_attention = None
            return samples, attention_maps
        return samples
