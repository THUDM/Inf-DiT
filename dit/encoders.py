import torch
from torch import nn
import torch.nn.functional as F

# import clip
import open_clip
import argparse

from sat import AutoModel
from sat.quantization.kernels import quantize
from transformers import AutoTokenizer
from torchvision import transforms
class ClipEncoder(nn.Module):
    def __init__(
        self,
        model='RN50x4',
        context_length=72,
        jit=False,
        antialias=False,
        input_resolution=288,
        noise_level=0,
        interpolate_when_encode_both=True,
        return_mask_when_encode_text=True,
        download_root=None,
        interpolate_when_encode_both_into_size=None,
        embedding_scale=1,
        dtype=torch.float32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.embedding_scale = embedding_scale
        self.encoder_dtype = dtype
        self.device = device
        self.model = self.model.to(device)
        
        self.antialias = antialias
        self.context_length = context_length
        self.interpolate_when_encode_both = interpolate_when_encode_both
        self.return_mask_when_encode_text = return_mask_when_encode_text
        self.interpolate_when_encode_both_into_size = interpolate_when_encode_both_into_size
        
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

        self.freeze()
        
    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def preprocess(self, x):
        import kornia
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (self.input_resolution, self.input_resolution),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x
    
    def forward(self, tokens):
        return self.encode(tokens)

    def encode(self, tokens):
        z, text_map, eos_pos = self.model.encode_text(tokens, output_features=True)
        text_map = text_map[:, :self.context_length, :]
        z = z / z.norm(dim=1, keepdim=True)
        text_map = text_map / text_map.norm(dim=-1, keepdim=True) * self.embedding_scale
        if self.return_mask_when_encode_text:
            mask = torch.arange(text_map.shape[1], device=text_map.device)
            mask = mask.expand(text_map.shape[0], -1) <= eos_pos[:, None]
            return text_map, mask[:, None, None, :]
        else:
            return text_map, None

    def tokenize(self, texts):
        tokens = clip.tokenize(texts, truncate=True, context_length=77).to(torch.int64)
        return {"tokens": tokens}

    def encode_text(self, texts):
        device = next(self.model.parameters()).device
        tokens = self.tokenize(texts).to(device)
        return self.encode(tokens)

class OpenClipEncoder(nn.Module):

    LAYERS = ["pooled", "last", "penultimate"]
    
    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        always_return_pooled=False,
        legacy=True,
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        del model.visual
        self.model = model
        self.encoder_dtype = next(self.parameters()).dtype

        self.device = device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = legacy

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, tokens):
        return self.encode(tokens)

    def encode(self, tokens):
        return self.encode_with_transformer(tokens)
    
    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            mask = torch.ones(*x.shape[:2])
            return x, mask[:, None, None, :]
        else:
            # x is a dict and will stay a dict
            o = x["last"]
            o = self.model.ln_final(o)
            pooled = self.pool(o, text)
            x["pooled"] = pooled
            return x

    def pool(self, x, text):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            @ self.model.text_projection
        )
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD
        return outputs

    def tokenize(self, texts):
        tokens = open_clip.tokenize(texts, context_length=self.max_length).to(torch.int64)
        return {"tokens": tokens}

    
class ChatGLMEncoder(nn.Module):
    def __init__(self, model_path='chatglm2-6b', tokenizer_path='chatglm2-6b', max_length=72, quant_bit=None, use_layer=-1, normalize=False, full_attention_mask=False, return_mask_when_encode_text=False, dtype=torch.float16, device='cuda'):
        super().__init__()
        self.max_length = max_length
        self.return_mask_when_encode_text = return_mask_when_encode_text
        self.use_layer = use_layer
        self.normalize = normalize
        self.full_attention_mask = full_attention_mask
        self.encoder_dtype = dtype
        self.device = device

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        model, model_args = AutoModel.from_pretrained(model_path, args=argparse.Namespace(
            fp16=True,
            skip_init=True,
            use_gpu_initialization=True,
        ))
        if quant_bit is not None:
            model.transformer = quantize(model.transformer, quant_bit).to(device)
        else:
            model.transformer = model.transformer.to(device)

        self.tokenizer = tokenizer
        self.model = model.eval()

    def forward(self, input_ids, attention_mask, position_ids):
        return self.encode(input_ids, attention_mask, position_ids)

    def tokenize(self, texts):
        batch_encoding = self.tokenizer(texts, 
            max_length=self.max_length, truncation=True,
            padding='max_length', return_tensors='pt')
        return batch_encoding

    def encode(self, input_ids, attention_mask, position_ids):
        if self.full_attention_mask:
            attention_mask = attention_mask[:, None, None, :]

        context = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )[self.use_layer]['hidden_states'].to(torch.float32)

        if self.normalize:
            context = context / context.norm(dim=-1, keepdim=True)
        if self.return_mask_when_encode_text:
            return context, attention_mask
        else:
            return context, None

    def encode_text(self, texts):
        device = next(self.model.parameters()).device
        batch_encoding = self.tokenizer(texts, 
            max_length=self.max_length, truncation=True,
            padding='max_length', return_tensors='pt')
        return self.encode(batch_encoding)


class OpenClipImageEncoder(nn.Module):

    def __init__(
            self,
            arch="ViT-L-14",
            version="datacomp_xl_s13b_b90k",
            device="cuda",
            freeze=True,
            image_size=224,
            cache_dir=None,
    ):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
            cache_dir=cache_dir,
        )

        del model.ln_final
        del model.transformer

        self.model = model
        self.encoder_dtype = next(self.parameters()).dtype
        self.device = device
        if freeze:
            self.freeze()
        self.preprocess = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image, add_text=False):
        image = self.preprocess(image)
        return self.encode(image, add_text=add_text)

    def tokenize(self, texts):
        tokens = open_clip.tokenize(texts, context_length=77).to(torch.int64)
        return tokens


    def encode(self, image, add_text=False):
        cls_embedding = self.model.encode_image(image, normalize=True)

        add_text = False
        # all neg_prompt = "noisy,blurry,low resolution."
        # HD HD photo
        # macro macro lens
        if add_text:
            scale = 2
            pos_prompt = "HD, high resolution, clear."
            neg_prompt = "noisy,blurry,low resolution."
            # macro lens photo
            batch_size = cls_embedding.shape[0]
            texts1 = [pos_prompt] * batch_size
            texts1 = self.tokenize(texts1).to(self.device)
            cls_embedding = cls_embedding + scale * self.model.encode_text(texts1, normalize=True)

            #low quality photo
            texts2 = [neg_prompt] * batch_size
            texts2 = self.tokenize(texts2).to(self.device)
            cls_embedding = cls_embedding - scale * self.model.encode_text(texts2, normalize=True)

        cls_embedding = F.normalize(cls_embedding, dim=-1)

        return cls_embedding
