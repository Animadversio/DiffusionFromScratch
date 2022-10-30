"""
Reconstructing UNet in Stable Diffusion from scratch in ~ 300 lines of codes

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import OrderedDict
from easydict import EasyDict as edict


class UNet_SD(nn.Module):
    def __init__(self, cat_unet=True):
        super().__init__()
        self.in_channels = 4
        self.out_channels = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_channels = 320
        time_proj_dim = 320
        time_emb_dim = 1280
        context_dim = 768
        nlevel = 4
        self.base_channels = base_channels
        attn_levels = [0, 1, 2]
        level_channels = [base_channels * mult for mult in [1, 2, 4, 4]]
        # Transform time into embedding
        self.time_embedding = nn.Sequential(OrderedDict({
            "linear_1": nn.Linear(time_proj_dim, time_emb_dim, bias=True),
            "act": nn.SiLU(),
            "linear_2": nn.Linear(time_emb_dim, time_emb_dim, bias=True),
        })
        )  # 2 layer MLP
        self.conv_in = nn.Conv2d(self.in_channels, base_channels, 3, stride=1, padding=1)

        # Tensor Downsample blocks
        nResAttn_block = 2
        self.down_blocks = TimeModulatedSequential()  # nn.ModuleList()
        self.down_blocks_channels = [base_channels]
        cur_chan = base_channels
        for i in range(nlevel):
            for j in range(nResAttn_block):
                res_attn_sandwich = TimeModulatedSequential()
                # input_chan of first ResBlock is different from the rest.
                res_attn_sandwich.append(ResBlock(in_channel=cur_chan, time_emb_dim=time_emb_dim, out_channel=level_channels[i]))
                if i in attn_levels:
                    # add attention except for the last level
                    res_attn_sandwich.append(SpatialTransformer(level_channels[i], context_dim=context_dim))
                cur_chan = level_channels[i]
                self.down_blocks.append(res_attn_sandwich)
                self.down_blocks_channels.append(cur_chan)
            # res_attn_sandwich.append(DownSample(level_channels[i]))
            if not i == nlevel - 1:
                self.down_blocks.append(TimeModulatedSequential(DownSample(level_channels[i])))
                self.down_blocks_channels.append(cur_chan)

        self.mid_block = TimeModulatedSequential(
            ResBlock(cur_chan, time_emb_dim),
            SpatialTransformer(cur_chan, context_dim=context_dim),
            ResBlock(cur_chan, time_emb_dim),
        )

        # Tensor Upsample blocks
        self.up_blocks = nn.ModuleList() # TimeModulatedSequential()  #
        for i in reversed(range(nlevel)):
            for j in range(nResAttn_block + 1):
                res_attn_sandwich = TimeModulatedSequential()
                res_attn_sandwich.append(ResBlock(in_channel=cur_chan + self.down_blocks_channels.pop(),
                                                  time_emb_dim=time_emb_dim, out_channel=level_channels[i]))
                if i in attn_levels:
                    res_attn_sandwich.append(SpatialTransformer(level_channels[i], context_dim=context_dim))
                cur_chan = level_channels[i]
                if j == nResAttn_block and i != 0:
                    res_attn_sandwich.append(UpSample(level_channels[i]))
                self.up_blocks.append(res_attn_sandwich)
        # Read out from tensor to latent space
        self.output = nn.Sequential(
            nn.GroupNorm(32, base_channels, ),
            nn.SiLU(),
            nn.Conv2d(base_channels, self.out_channels, 3, padding=1),
        )
        self.to(self.device)

    def time_proj(self, time_steps, max_period: int = 10000):
        if time_steps.ndim == 0:
            time_steps = time_steps.unsqueeze(0)
        half = self.base_channels // 2
        frequencies = torch.exp(- math.log(max_period)
             * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        angles = time_steps[:, None].float() * frequencies[None, :]
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    def forward(self, x, time_steps, cond=None, encoder_hidden_states=None, output_dict=True):
        if cond is None and encoder_hidden_states is not None:
            cond = encoder_hidden_states
        t_emb = self.time_proj(time_steps)
        t_emb = self.time_embedding(t_emb)
        x = self.conv_in(x)
        down_x_cache = [x]
        for module in self.down_blocks:
            x = module(x, t_emb, cond)
            down_x_cache.append(x)
        x = self.mid_block(x, t_emb, cond)
        for module in self.up_blocks:
            x = module(torch.cat((x, down_x_cache.pop()), dim=1), t_emb, cond)
        x = self.output(x)
        if output_dict:
            return edict(sample=x)
        else:
            return x

# Modified Container. Modify the nn.Sequential to time modulated Sequential
class TimeModulatedSequential(nn.Sequential):
    """ Modify the nn.Sequential to time modulated Sequential """
    def forward(self, x, t_emb, cond=None):
        for module in self:
            if isinstance(module, TimeModulatedSequential):
                x = module(x, t_emb, cond)
            elif isinstance(module, ResBlock):
                # For certain layers, add the time modulation.
                x = module(x, t_emb)
            elif isinstance(module, SpatialTransformer):
                # For certain layers, add the class conditioning.
                x = module(x, cond=cond)
            else:
                x = module(x)

        return x


# backbone, Residual Block (Checked)
class ResBlock(nn.Module):
    def __init__(self, in_channel, time_emb_dim, out_channel=None, ):
        super().__init__()
        if out_channel is None:
            out_channel = in_channel
        self.norm1 = nn.GroupNorm(32, in_channel, eps=1e-05, affine=True)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(in_features=time_emb_dim, out_features=out_channel, bias=True)
        self.norm2 = nn.GroupNorm(32, out_channel, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        if in_channel == out_channel:
            self.conv_shortcut = nn.Identity()
        else:
            self.conv_shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)

    def forward(self, x, t_emb, cond=None):
        # Input conv
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        # Time modulation
        if t_emb is not None:
            t_hidden = self.time_emb_proj(self.nonlinearity(t_emb))
            h = h + t_hidden[:, :, None, None]
        # Output conv
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        # Skip connection
        return h + self.conv_shortcut(x)


# UpSampling (Checked)
class UpSample(nn.Module):
    def __init__(self, channel, scale_factor=2, mode='nearest'):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


# DownSampling (Checked)
class DownSample(nn.Module):
    def __init__(self, channel, ):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, )

    def forward(self, x):
        return self.conv(x)  # F.interpolate(x, scale_factor=1/self.scale_factor, mode=self.mode)


# Transformer layers
# Self and Cross Attention mechanism (Checked)
class CrossAttention(nn.Module):
    """General implementation of Cross & Self Attention multi-head
    """
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=8, ):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.to_q = nn.Linear(hidden_dim, embed_dim, bias=False)
        if context_dim is None:
            # Self Attention
            self.to_k = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.to_v = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.self_attn = True
        else:
            # Cross Attention
            self.to_k = nn.Linear(context_dim, embed_dim, bias=False)
            self.to_v = nn.Linear(context_dim, embed_dim, bias=False)
            self.self_attn = False
        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=True)
        )  # this could be omitted

    def forward(self, tokens, context=None):
        Q = self.to_q(tokens)
        K = self.to_k(tokens) if self.self_attn else self.to_k(context)
        V = self.to_v(tokens) if self.self_attn else self.to_v(context)
        # print(Q.shape, K.shape, V.shape)
        # transform heads onto batch dimension
        Q = rearrange(Q, 'B T (H D) -> (B H) T D', H=self.num_heads, D=self.head_dim)
        K = rearrange(K, 'B T (H D) -> (B H) T D', H=self.num_heads, D=self.head_dim)
        V = rearrange(V, 'B T (H D) -> (B H) T D', H=self.num_heads, D=self.head_dim)
        # print(Q.shape, K.shape, V.shape)
        scoremats = torch.einsum("BTD,BSD->BTS", Q, K)
        attnmats = F.softmax(scoremats / math.sqrt(self.head_dim), dim=-1)
        # print(scoremats.shape, attnmats.shape, )
        ctx_vecs = torch.einsum("BTS,BSD->BTD", attnmats, V)
        # split the heads transform back to hidden.
        ctx_vecs = rearrange(ctx_vecs, '(B H) T D -> B T (H D)', H=self.num_heads, D=self.head_dim)
        # TODO: note this `to_out` is also a linear layer, could be in principle merged into the to_value layer.
        return self.to_out(ctx_vecs)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_heads=8):
        super(TransformerBlock, self).__init__()
        self.attn1 = CrossAttention(hidden_dim, hidden_dim, num_heads=num_heads)  # self attention
        self.attn2 = CrossAttention(hidden_dim, hidden_dim, context_dim, num_heads=num_heads)  # cross attention

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        # to be compatible with Diffuser, could simplify.
        self.ff = FeedForward_GEGLU(hidden_dim, )
        # self.ff = nn.Sequential(
        #     nn.Linear(hidden_dim, 3 * hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(3 * hidden_dim, hidden_dim)
        # )

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class GEGLU_proj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GEGLU_proj, self).__init__()
        self.proj = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x):
        x = self.proj(x)
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward_GEGLU(nn.Module):
    # https://github.com/huggingface/diffusers/blob/95414bd6bf9bb34a312a7c55f10ba9b379f33890/src/diffusers/models/attention.py#L339
    # A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.
    def __init__(self, hidden_dim, mult=4):
        super(FeedForward_GEGLU, self).__init__()
        self.net = nn.Sequential(
            GEGLU_proj(hidden_dim, mult * hidden_dim),
            nn.Dropout(0.0),
            nn.Linear(mult * hidden_dim, hidden_dim)
        )  # to be compatible with Diffuser, could simplify.

    def forward(self, x, ):
        return self.net(x)


class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_heads=8):
        super(SpatialTransformer, self).__init__()
        self.norm = nn.GroupNorm(32, hidden_dim, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        # self.transformer = TransformerBlock(hidden_dim, context_dim, num_heads=8)
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(hidden_dim, context_dim, num_heads=8)
        )  # to be compatible with Diffuser, could simplify.
        self.proj_out = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x, cond=None):
        b, c, h, w = x.shape
        x_in = x
        # context = rearrange(context, "b c T -> b T c")
        x = self.proj_in(self.norm(x))
        x = rearrange(x, "b c h w->b (h w) c")
        x = self.transformer_blocks[0](x, cond)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return self.proj_out(x) + x_in


def load_pipe_into_our_UNet(myUNet, pipe_unet):
    # load the pretrained weights from the pipe into our UNet.
    # Loading input and output layers.
    myUNet.output[0].load_state_dict(pipe_unet.conv_norm_out.state_dict())
    myUNet.output[2].load_state_dict(pipe_unet.conv_out.state_dict())
    myUNet.conv_in.load_state_dict(pipe_unet.conv_in.state_dict())
    myUNet.time_embedding.load_state_dict(pipe_unet.time_embedding.state_dict())
    #% Loading the down blocks
    myUNet.down_blocks[0][0].load_state_dict(pipe_unet.down_blocks[0].resnets[0].state_dict())
    myUNet.down_blocks[0][1].load_state_dict(pipe_unet.down_blocks[0].attentions[0].state_dict())
    myUNet.down_blocks[1][0].load_state_dict(pipe_unet.down_blocks[0].resnets[1].state_dict())
    myUNet.down_blocks[1][1].load_state_dict(pipe_unet.down_blocks[0].attentions[1].state_dict())
    myUNet.down_blocks[2][0].load_state_dict(pipe_unet.down_blocks[0].downsamplers[0].state_dict())

    myUNet.down_blocks[3][0].load_state_dict(pipe_unet.down_blocks[1].resnets[0].state_dict())
    myUNet.down_blocks[3][1].load_state_dict(pipe_unet.down_blocks[1].attentions[0].state_dict())
    myUNet.down_blocks[4][0].load_state_dict(pipe_unet.down_blocks[1].resnets[1].state_dict())
    myUNet.down_blocks[4][1].load_state_dict(pipe_unet.down_blocks[1].attentions[1].state_dict())
    myUNet.down_blocks[5][0].load_state_dict(pipe_unet.down_blocks[1].downsamplers[0].state_dict())

    myUNet.down_blocks[6][0].load_state_dict(pipe_unet.down_blocks[2].resnets[0].state_dict())
    myUNet.down_blocks[6][1].load_state_dict(pipe_unet.down_blocks[2].attentions[0].state_dict())
    myUNet.down_blocks[7][0].load_state_dict(pipe_unet.down_blocks[2].resnets[1].state_dict())
    myUNet.down_blocks[7][1].load_state_dict(pipe_unet.down_blocks[2].attentions[1].state_dict())
    myUNet.down_blocks[8][0].load_state_dict(pipe_unet.down_blocks[2].downsamplers[0].state_dict())

    myUNet.down_blocks[9][0].load_state_dict(pipe_unet.down_blocks[3].resnets[0].state_dict())
    myUNet.down_blocks[10][0].load_state_dict(pipe_unet.down_blocks[3].resnets[1].state_dict())

    #% Loading the middle blocks
    myUNet.mid_block[0].load_state_dict(pipe_unet.mid_block.resnets[0].state_dict())
    myUNet.mid_block[1].load_state_dict(pipe_unet.mid_block.attentions[0].state_dict())
    myUNet.mid_block[2].load_state_dict(pipe_unet.mid_block.resnets[1].state_dict())
    # % Loading the up blocks
    # upblock 0
    myUNet.up_blocks[0][0].load_state_dict(pipe_unet.up_blocks[0].resnets[0].state_dict())
    myUNet.up_blocks[1][0].load_state_dict(pipe_unet.up_blocks[0].resnets[1].state_dict())
    myUNet.up_blocks[2][0].load_state_dict(pipe_unet.up_blocks[0].resnets[2].state_dict())
    myUNet.up_blocks[2][1].load_state_dict(pipe_unet.up_blocks[0].upsamplers[0].state_dict())
    # % upblock 1
    myUNet.up_blocks[3][0].load_state_dict(pipe_unet.up_blocks[1].resnets[0].state_dict())
    myUNet.up_blocks[3][1].load_state_dict(pipe_unet.up_blocks[1].attentions[0].state_dict())
    myUNet.up_blocks[4][0].load_state_dict(pipe_unet.up_blocks[1].resnets[1].state_dict())
    myUNet.up_blocks[4][1].load_state_dict(pipe_unet.up_blocks[1].attentions[1].state_dict())
    myUNet.up_blocks[5][0].load_state_dict(pipe_unet.up_blocks[1].resnets[2].state_dict())
    myUNet.up_blocks[5][1].load_state_dict(pipe_unet.up_blocks[1].attentions[2].state_dict())
    myUNet.up_blocks[5][2].load_state_dict(pipe_unet.up_blocks[1].upsamplers[0].state_dict())
    # % upblock 2
    myUNet.up_blocks[6][0].load_state_dict(pipe_unet.up_blocks[2].resnets[0].state_dict())
    myUNet.up_blocks[6][1].load_state_dict(pipe_unet.up_blocks[2].attentions[0].state_dict())
    myUNet.up_blocks[7][0].load_state_dict(pipe_unet.up_blocks[2].resnets[1].state_dict())
    myUNet.up_blocks[7][1].load_state_dict(pipe_unet.up_blocks[2].attentions[1].state_dict())
    myUNet.up_blocks[8][0].load_state_dict(pipe_unet.up_blocks[2].resnets[2].state_dict())
    myUNet.up_blocks[8][1].load_state_dict(pipe_unet.up_blocks[2].attentions[2].state_dict())
    myUNet.up_blocks[8][2].load_state_dict(pipe_unet.up_blocks[2].upsamplers[0].state_dict())
    # % upblock 3
    myUNet.up_blocks[9][0].load_state_dict(pipe_unet.up_blocks[3].resnets[0].state_dict())
    myUNet.up_blocks[9][1].load_state_dict(pipe_unet.up_blocks[3].attentions[0].state_dict())
    myUNet.up_blocks[10][0].load_state_dict(pipe_unet.up_blocks[3].resnets[1].state_dict())
    myUNet.up_blocks[10][1].load_state_dict(pipe_unet.up_blocks[3].attentions[1].state_dict())
    myUNet.up_blocks[11][0].load_state_dict(pipe_unet.up_blocks[3].resnets[2].state_dict())
    myUNet.up_blocks[11][1].load_state_dict(pipe_unet.up_blocks[3].attentions[2].state_dict())