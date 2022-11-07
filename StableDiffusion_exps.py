"""Experimenting with StableDiffusion and our version. """

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt


def plt_show_image(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def recursive_print(module, prefix="", depth=0, deepest=3):
    """Simulating print(module) for torch.nn.Modules
        but with depth control. Print to the `deepest` level. `deepest=0` means no print
    """
    if depth >= deepest:
        return
    for name, child in module.named_children():
        if len([*child.named_children()]) == 0:
            print(f"{prefix}({name}): {child}")
        else:
            print(f"{prefix}({name}): {type(child).__name__}")
        recursive_print(child, prefix + "  ", depth + 1, deepest)

#%%

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True
).to("cuda")
def dummy_checker(images, **kwargs): return images, False
pipe.safety_checker = dummy_checker
#%%
recursive_print(pipe.unet, deepest=2)
#%% Text to
# prompt = "a photo of an ballerina riding a horse on mars"
prompt = "A ballerina riding a Harley Motorcycle, CG Art"
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]

image.save("astronaut_rides_horse.png")
plt_show_image(image)
#%% Loading in our own model!
from StableDiff_UNet_model import UNet_SD, load_pipe_into_our_UNet
myunet = UNet_SD()
original_unet = pipe.unet.cpu()
load_pipe_into_our_UNet(myunet, original_unet)
pipe.unet = myunet.cuda()
#%%
torch.save(myunet.state_dict(), "/home/binxuwang/DL_Projects/SDfromScratch/ourUNet.pth")


#%% Saving images during diffusion process using callback

latents_reservoir = []
@torch.no_grad()
def plot_show_callback(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())
    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    plt_show_image(image[0])
    plt.imsave(f"/home/binxuwang/DL_Projects/SDfromScratch/diffproc/sample_{i:02d}.png", image[0])

latents_reservoir = []
@torch.no_grad()
def save_latents(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())
#%%
# prompt = "A ballerina dancing on a high ground in the starry night"
# prompt = "A cute cat running on the grass in the style of Monet"
prompt = "A ballerina chasing her cat running on the grass in the style of Monet"
prompt = "A kitty cat dressed like Lincoln, old timey style"
with autocast("cuda"):
    image = pipe(prompt, callback=plot_show_callback)["sample"][0]  # plot_show_callback

image.save("cat_Lincoln.png")
plt_show_image(image)
#%%
len(latents_reservoir)
plt_show_image(latents_reservoir[-1][0, [0, 1, 2,], :].permute(1, 2, 0).cpu().numpy() / 1.6 + 0.4)
#%% Visualize architecture

#%% Full unets
recursive_print(pipe.unet, deepest=3)
#%%
recursive_print(pipe.vae, deepest=3)
#%% Down blocks
recursive_print(pipe.unet.down_blocks, deepest=4)
#%% Up blocks
recursive_print(pipe.unet.up_blocks, deepest=4)
#%%
torch.save(pipe.unet.state_dict(), "/home/binxuwang/DL_Projects/SDfromScratch/SD_unet.pt",)
torch.save(pipe.vae.state_dict(), "/home/binxuwang/DL_Projects/SDfromScratch/SD_vae.pt",)
#%%
SD_unet = torch.load("/home/binxuwang/DL_Projects/SDfromScratch/SD_unet.pt")
#%%
# https://github.com/CompVis/stable-diffusion/blob/main/configs/stable-diffusion/v1-inference.yaml