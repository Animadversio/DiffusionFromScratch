"""
Test out the key modules for Stable Diffusion
 - ResBlock
 - UpSample
 - DownSample
 - CrossAttention
 - BasicTransformer (self, cross, FFN)
 - Spatial Transformer
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from StableDiff_UNet_model import *

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True
).to("cuda")
#%% test ResBlock Implementation
tmp_blk = ResBlock(320, 1280).cuda()
std_blk = pipe.unet.down_blocks[0].resnets[0]
SD = std_blk.state_dict()
tmp_blk.load_state_dict(SD)
lat_tmp = torch.randn(3, 320, 32, 32)
temb = torch.randn(3, 1280)
with torch.no_grad():
    out = pipe.unet.down_blocks[0].resnets[0](lat_tmp.cuda(),temb.cuda())
    out2 = tmp_blk(lat_tmp.cuda(), temb.cuda())

assert torch.allclose(out2, out)

#%% test downsampler
tmpdsp = DownSample(320).cuda()
stddsp = pipe.unet.down_blocks[0].downsamplers[0]
SD = stddsp.state_dict()
tmpdsp.load_state_dict(SD)
lat_tmp = torch.randn(3, 320, 32, 32)
with torch.no_grad():
    out = stddsp(lat_tmp.cuda())
    out2 = tmpdsp(lat_tmp.cuda())

assert torch.allclose(out2, out)

#%% test upsampler
tmpusp = UpSample(1280).cuda()
stdusp = pipe.unet.up_blocks[1].upsamplers[0]
SD = stdusp.state_dict()
tmpusp.load_state_dict(SD)
lat_tmp = torch.randn(3, 1280, 32, 32)
with torch.no_grad():
    out = stdusp(lat_tmp.cuda())
    out2 = tmpusp(lat_tmp.cuda())

assert torch.allclose(out2, out)


#%% test SpatialTransformer Implementation
# Check self attention
tmpSattn = CrossAttention(320, 320, context_dim=None, num_heads=8).cuda()
stdSattn = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1
tmpSattn.load_state_dict(stdSattn.state_dict())  # checked
with torch.no_grad():
    lat_tmp = torch.randn(3, 32, 320)
    out = stdSattn(lat_tmp.cuda())
    out2 = tmpSattn(lat_tmp.cuda())
assert torch.allclose(out2, out)  # False

#%%
# Check Cross attention
tmpXattn = CrossAttention(320, 320, context_dim=768, num_heads=8).cuda()
stdXattn = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2
tmpXattn.load_state_dict(stdXattn.state_dict())  # checked
with torch.no_grad():
    lat_tmp = torch.randn(3, 32, 320)
    ctx_tmp = torch.randn(3, 5, 768)
    out = stdXattn(lat_tmp.cuda(), ctx_tmp.cuda())
    out2 = tmpXattn(lat_tmp.cuda(), ctx_tmp.cuda())
assert torch.allclose(out2, out)  # False

#%% test TransformerBlock Implementation
tmpTfmer = TransformerBlock(320, context_dim=768, num_heads=8).cuda()
stdTfmer = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0]
tmpTfmer.load_state_dict(stdTfmer.state_dict())  # checked
with torch.no_grad():
    lat_tmp = torch.randn(3, 32, 320)
    ctx_tmp = torch.randn(3, 5, 768)
    out = tmpTfmer(lat_tmp.cuda(), ctx_tmp.cuda())
    out2 = stdTfmer(lat_tmp.cuda(), ctx_tmp.cuda())
assert torch.allclose(out2, out)  # False


#%% test SpatialTransformer Implementation
tmpSpTfmer = SpatialTransformer(320, context_dim=768, num_heads=8).cuda()
stdSpTfmer = pipe.unet.down_blocks[0].attentions[0]
tmpSpTfmer.load_state_dict(stdSpTfmer.state_dict())  # checked
with torch.no_grad():
    lat_tmp = torch.randn(3, 320, 8, 8)
    ctx_tmp = torch.randn(3, 5, 768)
    out = tmpSpTfmer(lat_tmp.cuda(), ctx_tmp.cuda())
    out2 = stdSpTfmer(lat_tmp.cuda(), ctx_tmp.cuda())
assert torch.allclose(out2, out)  # False

#%% test UNet downblocks
tmpUNet = UNet_SD()
load_pipe_into_our_UNet(tmpUNet, pipe)

#%%
tmpUNet.output[0].load_state_dict(pipe.unet.conv_norm_out.state_dict())
tmpUNet.output[2].load_state_dict(pipe.unet.conv_out.state_dict())
tmpUNet.conv_in.load_state_dict(pipe.unet.conv_in.state_dict())
tmpUNet.time_embedding.load_state_dict(pipe.unet.time_embedding.state_dict())

# Loading the down blocks
tmpUNet.down_blocks[0][0].load_state_dict(pipe.unet.down_blocks[0].resnets[0].state_dict())
tmpUNet.down_blocks[0][1].load_state_dict(pipe.unet.down_blocks[0].attentions[0].state_dict())
tmpUNet.down_blocks[1][0].load_state_dict(pipe.unet.down_blocks[0].resnets[1].state_dict())
tmpUNet.down_blocks[1][1].load_state_dict(pipe.unet.down_blocks[0].attentions[1].state_dict())
tmpUNet.down_blocks[2][0].load_state_dict(pipe.unet.down_blocks[0].downsamplers[0].state_dict())

tmpUNet.down_blocks[3][0].load_state_dict(pipe.unet.down_blocks[1].resnets[0].state_dict())
tmpUNet.down_blocks[3][1].load_state_dict(pipe.unet.down_blocks[1].attentions[0].state_dict())
tmpUNet.down_blocks[4][0].load_state_dict(pipe.unet.down_blocks[1].resnets[1].state_dict())
tmpUNet.down_blocks[4][1].load_state_dict(pipe.unet.down_blocks[1].attentions[1].state_dict())
tmpUNet.down_blocks[5][0].load_state_dict(pipe.unet.down_blocks[1].downsamplers[0].state_dict())

tmpUNet.down_blocks[6][0].load_state_dict(pipe.unet.down_blocks[2].resnets[0].state_dict())
tmpUNet.down_blocks[6][1].load_state_dict(pipe.unet.down_blocks[2].attentions[0].state_dict())
tmpUNet.down_blocks[7][0].load_state_dict(pipe.unet.down_blocks[2].resnets[1].state_dict())
tmpUNet.down_blocks[7][1].load_state_dict(pipe.unet.down_blocks[2].attentions[1].state_dict())
tmpUNet.down_blocks[8][0].load_state_dict(pipe.unet.down_blocks[2].downsamplers[0].state_dict())

tmpUNet.down_blocks[9][0].load_state_dict(pipe.unet.down_blocks[3].resnets[0].state_dict())
tmpUNet.down_blocks[10][0].load_state_dict(pipe.unet.down_blocks[3].resnets[1].state_dict())

# Loading the middle blocks
tmpUNet.mid_block[0].load_state_dict(pipe.unet.mid_block.resnets[0].state_dict())
tmpUNet.mid_block[1].load_state_dict(pipe.unet.mid_block.attentions[0].state_dict())
tmpUNet.mid_block[2].load_state_dict(pipe.unet.mid_block.resnets[1].state_dict())

#%% Loading the up blocks
# upblock 0
tmpUNet.up_blocks[0][0].load_state_dict(pipe.unet.up_blocks[0].resnets[0].state_dict())
tmpUNet.up_blocks[1][0].load_state_dict(pipe.unet.up_blocks[0].resnets[1].state_dict())
tmpUNet.up_blocks[2][0].load_state_dict(pipe.unet.up_blocks[0].resnets[2].state_dict())
tmpUNet.up_blocks[2][1].load_state_dict(pipe.unet.up_blocks[0].upsamplers[0].state_dict())
#% upblock 1
tmpUNet.up_blocks[3][0].load_state_dict(pipe.unet.up_blocks[1].resnets[0].state_dict())
tmpUNet.up_blocks[3][1].load_state_dict(pipe.unet.up_blocks[1].attentions[0].state_dict())
tmpUNet.up_blocks[4][0].load_state_dict(pipe.unet.up_blocks[1].resnets[1].state_dict())
tmpUNet.up_blocks[4][1].load_state_dict(pipe.unet.up_blocks[1].attentions[1].state_dict())
tmpUNet.up_blocks[5][0].load_state_dict(pipe.unet.up_blocks[1].resnets[2].state_dict())
tmpUNet.up_blocks[5][1].load_state_dict(pipe.unet.up_blocks[1].attentions[2].state_dict())
tmpUNet.up_blocks[5][2].load_state_dict(pipe.unet.up_blocks[1].upsamplers[0].state_dict())
#% upblock 2
tmpUNet.up_blocks[6][0].load_state_dict(pipe.unet.up_blocks[2].resnets[0].state_dict())
tmpUNet.up_blocks[6][1].load_state_dict(pipe.unet.up_blocks[2].attentions[0].state_dict())
tmpUNet.up_blocks[7][0].load_state_dict(pipe.unet.up_blocks[2].resnets[1].state_dict())
tmpUNet.up_blocks[7][1].load_state_dict(pipe.unet.up_blocks[2].attentions[1].state_dict())
tmpUNet.up_blocks[8][0].load_state_dict(pipe.unet.up_blocks[2].resnets[2].state_dict())
tmpUNet.up_blocks[8][1].load_state_dict(pipe.unet.up_blocks[2].attentions[2].state_dict())
tmpUNet.up_blocks[8][2].load_state_dict(pipe.unet.up_blocks[2].upsamplers[0].state_dict())
#% upblock 3
tmpUNet.up_blocks[9][0].load_state_dict(pipe.unet.up_blocks[3].resnets[0].state_dict())
tmpUNet.up_blocks[9][1].load_state_dict(pipe.unet.up_blocks[3].attentions[0].state_dict())
tmpUNet.up_blocks[10][0].load_state_dict(pipe.unet.up_blocks[3].resnets[1].state_dict())
tmpUNet.up_blocks[10][1].load_state_dict(pipe.unet.up_blocks[3].attentions[1].state_dict())
tmpUNet.up_blocks[11][0].load_state_dict(pipe.unet.up_blocks[3].resnets[2].state_dict())
tmpUNet.up_blocks[11][1].load_state_dict(pipe.unet.up_blocks[3].attentions[2].state_dict())
#%%

#%% Check entire UNet, very small difference
tmpUNet.cuda().eval()
pipe.unet.eval()
with torch.no_grad():
    lat_x = torch.randn(3, 4, 32, 32).cuda()
    ctx_tmp = torch.randn(3, 5, 768).cuda()
    t_emb_tmp = torch.rand(3, ).cuda()
    out = tmpUNet(lat_x, t_emb_tmp, ctx_tmp)
    out2 = pipe.unet(lat_x, t_emb_tmp, ctx_tmp)
    sample = out2.sample

print((sample - out).max(), (sample - out).min())  # 0.0008 -0.0008
assert torch.allclose(sample, out)  # False
#%% checked all downward blocks [Checked]
tmpUNet.mid_block.cuda()
tmpUNet.down_blocks.cuda()
with torch.no_grad():
    lat_tmp = torch.randn(3, 320, 32, 32)
    ctx_tmp = torch.randn(3, 5, 768)
    t_emb_tmp = torch.randn(3, 1280)
    out = tmpUNet.down_blocks[0:3](lat_tmp.cuda(), t_emb_tmp.cuda(), ctx_tmp.cuda())
    out2 = pipe.unet.down_blocks[0](lat_tmp.cuda(), t_emb_tmp.cuda(), ctx_tmp.cuda())
assert torch.allclose(out2[0], out)  # False

#%% checked all downward and middle blocks [Checked]
with torch.no_grad():
    lat_tmp = torch.randn(3, 320, 64, 64)
    ctx_tmp = torch.randn(3, 5, 768)
    t_emb_tmp = torch.randn(3, 1280)
    # our implementation
    downout = tmpUNet.down_blocks(lat_tmp.cuda(), t_emb_tmp.cuda(), ctx_tmp.cuda())
    out = tmpUNet.mid_block(downout, t_emb_tmp.cuda(), ctx_tmp.cuda())
    # standard implementation
    hidden = lat_tmp.cuda()
    for i in range(3):
        hidden, out_col = pipe.unet.down_blocks[i](hidden, t_emb_tmp.cuda(), ctx_tmp.cuda())
    downout2, out_col = pipe.unet.down_blocks[-1](hidden, t_emb_tmp.cuda(), )
    out2 = pipe.unet.mid_block(downout2, t_emb_tmp.cuda(), ctx_tmp.cuda())
assert torch.allclose(out2, out)  # False
assert torch.allclose(downout2, downout)  # False

#%% checked all downward blocks [Checked, exact]
tmpUNet.mid_block.cuda()
tmpUNet.down_blocks.cuda()
tmpUNet.up_blocks.cuda()
with torch.no_grad():
    lat_tmp = torch.randn(2, 320, 32, 32).cuda()
    ctx_tmp = torch.randn(2, 5, 768).cuda()
    t_emb_tmp = torch.randn(2, 1280).cuda()
    # our implementation
    hidden = lat_tmp.cuda()
    down_x_cache = [hidden]
    for module in tmpUNet.down_blocks:
        hidden = module(hidden, t_emb_tmp, ctx_tmp)
        down_x_cache.append(hidden)
    out = tmpUNet.mid_block(hidden, t_emb_tmp, ctx_tmp)

    # Hugginface standard implementation
    hidden = lat_tmp.cuda()
    out_cache = (hidden, )
    for i in range(3):
        hidden, out_col = pipe.unet.down_blocks[i](hidden, t_emb_tmp, ctx_tmp)
        out_cache = out_cache + out_col
    downout2, out_col = pipe.unet.down_blocks[-1](hidden, t_emb_tmp,)
    out_cache = out_cache + out_col
    out2 = pipe.unet.mid_block(downout2, t_emb_tmp, ctx_tmp)

assert torch.allclose(out2, out)  # False
for x1, x2 in zip(down_x_cache, out_cache):
    assert torch.allclose(x1, x2)  # False
#%% checked all downward and middle and upward blocks [Checked, not exactly same!]
tmpUNet.mid_block.cuda()
tmpUNet.down_blocks.cuda()
tmpUNet.up_blocks.cuda()
tmpUNet.eval()
pipe.unet.eval()
with torch.no_grad():
    lat_tmp = torch.randn(2, 320, 32, 32).cuda()
    ctx_tmp = torch.randn(2, 5, 768).cuda()
    t_emb_tmp = torch.randn(2, 1280).cuda()
    # our implementation
    hidden = lat_tmp.cuda()
    down_x_cache = [hidden]
    for module in tmpUNet.down_blocks:
        hidden = module(hidden, t_emb_tmp, ctx_tmp)
        down_x_cache.append(hidden)
    out = tmpUNet.mid_block(hidden, t_emb_tmp, ctx_tmp)
    for module in tmpUNet.up_blocks[:]:
        out = module(torch.cat((out, down_x_cache.pop()), dim=1), t_emb_tmp, ctx_tmp)

    # Hugginface standard implementation
    hidden = lat_tmp.cuda()
    out_cache = (hidden, )
    for i in range(3):
        hidden, out_col = pipe.unet.down_blocks[i](hidden, t_emb_tmp, ctx_tmp)
        out_cache = out_cache + out_col
    downout2, out_col = pipe.unet.down_blocks[-1](hidden, t_emb_tmp,)
    out_cache = out_cache + out_col
    out2 = pipe.unet.mid_block(downout2, t_emb_tmp, ctx_tmp)
    out2 = pipe.unet.up_blocks[0](hidden_states=out2, temb=t_emb_tmp,
                res_hidden_states_tuple=out_cache[-3:],
    )
    out_cache = out_cache[:-3]
    for i in range(1, 4):
        out2 = pipe.unet.up_blocks[i](hidden_states=out2, temb=t_emb_tmp,
                    res_hidden_states_tuple=out_cache[-3:],
                    encoder_hidden_states=ctx_tmp,
        )
        out_cache = out_cache[:-3]
    # for i, upsample_block in enumerate(pipe.unet.up_blocks):
    #     is_final_block = i == len(pipe.unet.up_blocks) - 1
    #
    #     res_samples = out_cache[-len(upsample_block.resnets):]
    #     out_cache = out_cache[: -len(upsample_block.resnets)]
    #
    #     if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
    #         out2 = upsample_block(
    #             hidden_states=out2, temb=t_emb_tmp,
    #             res_hidden_states_tuple=res_samples,
    #             encoder_hidden_states=ctx_tmp,
    #         )
    #     else:
    #         out2 = upsample_block(
    #             hidden_states=out2, temb=t_emb_tmp,
    #             res_hidden_states_tuple=res_samples,
    #         )

print((out2 - out).max(), (out2 - out).min())
assert torch.allclose(out2, out)  # False
#%%

#%% Older version
# Load in down blocks
tmpDown = tmpUNet.down_blocks[0].cuda()
stdDown = pipe.unet.down_blocks[0]
tmpDown[0].load_state_dict(stdDown.resnets[0].state_dict())
tmpDown[1].load_state_dict(stdDown.attentions[0].state_dict())
tmpDown[2].load_state_dict(stdDown.resnets[1].state_dict())
tmpDown[3].load_state_dict(stdDown.attentions[1].state_dict())
tmpDown[4].load_state_dict(stdDown.downsamplers[0].state_dict())
tmpDown = tmpUNet.down_blocks[1].cuda()
stdDown = pipe.unet.down_blocks[1]
tmpDown[0].load_state_dict(stdDown.resnets[0].state_dict())
tmpDown[1].load_state_dict(stdDown.attentions[0].state_dict())
tmpDown[2].load_state_dict(stdDown.resnets[1].state_dict())
tmpDown[3].load_state_dict(stdDown.attentions[1].state_dict())
tmpDown[4].load_state_dict(stdDown.downsamplers[0].state_dict())
tmpDown = tmpUNet.down_blocks[2].cuda()
stdDown = pipe.unet.down_blocks[2]
tmpDown[0].load_state_dict(stdDown.resnets[0].state_dict())
tmpDown[1].load_state_dict(stdDown.attentions[0].state_dict())
tmpDown[2].load_state_dict(stdDown.resnets[1].state_dict())
tmpDown[3].load_state_dict(stdDown.attentions[1].state_dict())
tmpDown[4].load_state_dict(stdDown.downsamplers[0].state_dict())
tmpDown = tmpUNet.down_blocks[3].cuda()
stdDown = pipe.unet.down_blocks[3]
tmpDown[0].load_state_dict(stdDown.resnets[0].state_dict())
tmpDown[1].load_state_dict(stdDown.resnets[1].state_dict())
# Load in middle blocks
stdMid = pipe.unet.mid_block
tmpUNet.mid_block[0].load_state_dict(stdMid.resnets[0].state_dict())
tmpUNet.mid_block[1].load_state_dict(stdMid.attentions[0].state_dict())
tmpUNet.mid_block[2].load_state_dict(stdMid.resnets[1].state_dict())
#%% Check down sample blocks
with torch.no_grad():
    lat_tmp = torch.randn(3, 320, 32, 32)
    ctx_tmp = torch.randn(3, 5, 768)
    t_emb_tmp = torch.randn(3, 1280)
    out = tmpUNet.down_blocks[0](lat_tmp.cuda(), t_emb_tmp.cuda(), ctx_tmp.cuda())
    out2 = pipe.unet.down_blocks[0](lat_tmp.cuda(), t_emb_tmp.cuda(), ctx_tmp.cuda())
assert torch.allclose(out2[0], out)  # False
#%% Check down sample blocks
tmpUNet.down_blocks[1].cuda()
with torch.no_grad():
    lat_tmp = torch.randn(3, 320, 16, 16)
    ctx_tmp = torch.randn(3, 5, 768)
    t_emb_tmp = torch.randn(3, 1280)
    out = tmpUNet.down_blocks[1](lat_tmp.cuda(), t_emb_tmp.cuda(), ctx_tmp.cuda())
    out2 = pipe.unet.down_blocks[1](lat_tmp.cuda(), t_emb_tmp.cuda(), ctx_tmp.cuda())
assert torch.allclose(out2[0], out)  # False
#%% Down block 2
tmpUNet.down_blocks[2].cuda()
with torch.no_grad():
    lat_tmp = torch.randn(3, 640, 8, 8)
    ctx_tmp = torch.randn(3, 5, 768)
    t_emb_tmp = torch.randn(3, 1280)
    out = tmpUNet.down_blocks[2](lat_tmp.cuda(), t_emb_tmp.cuda(), ctx_tmp.cuda())
    out2 = pipe.unet.down_blocks[2](lat_tmp.cuda(), t_emb_tmp.cuda(), ctx_tmp.cuda())
assert torch.allclose(out2[0], out)  # False
#%% Down block 3
tmpUNet.down_blocks[3].cuda()
with torch.no_grad():
    lat_tmp = torch.randn(3, 1280, 8, 8)
    ctx_tmp = torch.randn(3, 5, 768)
    t_emb_tmp = torch.randn(3, 1280)
    out = tmpUNet.down_blocks[3](lat_tmp.cuda(), t_emb_tmp.cuda(), ctx_tmp.cuda())
    out2 = pipe.unet.down_blocks[3](lat_tmp.cuda(), t_emb_tmp.cuda())
assert torch.allclose(out2[0], out)  # False
#%% Check middle blocks
stdMid = pipe.unet.mid_block
tmpUNet.mid_block.cuda()
with torch.no_grad():
    lat_tmp = torch.randn(3, 1280, 8, 8)
    t_emb_tmp = torch.randn(3, 1280)
    ctx_tmp = torch.randn(3, 5, 768)
    out = tmpUNet.mid_block(lat_tmp.cuda(), t_emb_tmp.cuda(), ctx_tmp.cuda())
    out2 = stdMid(lat_tmp.cuda(), t_emb_tmp.cuda(), ctx_tmp.cuda())
assert torch.allclose(out2, out)  # False
#%% test UNet in total

