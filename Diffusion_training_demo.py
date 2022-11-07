import torch
import functools
from tqdm import tqdm, trange
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# @title Diffusion constant and noise strength
device = 'cuda'  # @param ['cuda', 'cpu'] {'type':'string'}

def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(sigma ** t, device=device)


sigma = 25.0  # @param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
#%%
#@title Training Loss function
def loss_fn_cond(model, x, y, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t, y=y)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss
#%%
# @title Diffusion Model Sampler
def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           x_shape=(1, 28, 28),
                           num_steps=500,
                           device='cuda',
                           eps=1e-3,
                           y=None):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, *x_shape, device=device) \
             * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step, y=y) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
            # Do not include any noise in the last sampling step.
    return mean_x

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import math
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from tqdm.notebook import trange, tqdm
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR

import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from einops import rearrange

#@title Diffusion Trainer
def train_score_model(score_model, dataset, lr, n_epochs, batch_size, ckpt_name,
                      marginal_prob_std_fn=marginal_prob_std_fn,
                      lr_scheduler_fn=lambda epoch: max(0.2, 0.98 ** epoch),
                      device="cuda",
                      callback=None): # resume=False,
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

  optimizer = Adam(score_model.parameters(), lr=lr)
  scheduler = LambdaLR(optimizer, lr_lambda=lr_scheduler_fn)
  tqdm_epoch = trange(n_epochs)
  for epoch in tqdm_epoch:
    score_model.train()
    avg_loss = 0.
    num_items = 0
    for x, y in tqdm(data_loader):
      x = x.to(device)
      loss = loss_fn_cond(score_model, x, y, marginal_prob_std_fn)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      avg_loss += loss.item() * x.shape[0]
      num_items += x.shape[0]
    scheduler.step()
    lr_current = scheduler.get_last_lr()[0]
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    # Print the averaged training loss so far.
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Update the checkpoint after each epoch of training.
    torch.save(score_model.state_dict(), f'ckpt_{ckpt_name}.pth')
    if callback is not None:
      score_model.eval()
      callback(score_model, epoch, ckpt_name)

#%%

#%%
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, CenterCrop, Resize, Compose, Normalize

tfm = Compose([
    Resize(32),
    CenterCrop(32),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset_rsz = CelebA("/home/binxuwang/Datasets", target_type=["attr"],
                    transform=tfm, download=False) # ,"identity"
#%%
# def preprocess_dataset(dataset_rsz, ):
dataloader = DataLoader(dataset_rsz, batch_size=64, num_workers=8, shuffle=False)
x_col = []
y_col = []
for xs, ys in tqdm(dataloader):
  x_col.append(xs)
  y_col.append(ys)
x_col = torch.concat(x_col, dim=0)
y_col = torch.concat(y_col, dim=0)
print(x_col.shape)
print(y_col.shape)

nantoken = 40
maxlen = (y_col.sum(dim=1)).max()
yseq_data = torch.ones(y_col.size(0), maxlen, dtype=int).fill_(nantoken)

saved_dataset = TensorDataset(x_col, yseq_data)
# return saved_dataset
#%%
import matplotlib.pyplot as plt

def save_sample_callback(score_model, epocs, ckpt_name):
    sample_batch_size = 64
    num_steps = 250
    y_samp = yseq_data[:sample_batch_size, :]
    sampler = Euler_Maruyama_sampler
    samples = sampler(score_model,
                      marginal_prob_std_fn,
                      diffusion_coeff_fn,
                      sample_batch_size,
                      x_shape=(3, 32, 32),
                      num_steps=num_steps,
                      device=device,
                      y=y_samp, )
    denormalize = Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        [1/0.229, 1/0.224, 1/0.225])
    samples = denormalize(samples).clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.tight_layout()
    plt.savefig(f"samples_{ckpt_name}_{epocs}.png")
    plt.show()
#%%
#@title Training model

# continue_training = False #@param {type:"boolean"}
# if not continue_training:
#   print("initilize new score model...")
score_model = torch.nn.DataParallel(
  UNet_Tranformer_attrb(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

n_epochs =   50
batch_size = 1024
lr = 10e-4
train_score_model(score_model, saved_dataset, lr, n_epochs, batch_size,
                  "Unet-tfmer_pad", device="cuda", callback=save_sample_callback)
#%%

#%%
n_epochs = 50
batch_size = 1048
lr = 2e-4
train_score_model(score_model, saved_dataset, lr, n_epochs, batch_size,
                  "Unet-tfmer", device="cuda", callback=save_sample_callback)
#%%

sample_batch_size = 64  #@param {'type':'integer'}
num_steps = 250  #@param {'type':'integer'}
sampler = Euler_Maruyama_sampler  #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}
# score_model.eval()
## Generate samples using the specified sampler.
samples = sampler(score_model,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        sample_batch_size,
        x_shape=(3, 32, 32),
        num_steps=num_steps,
        device=device,
        y=yseq_data[:sample_batch_size,:], )

## Sample visualization.
denormalize = Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        [1/0.229, 1/0.224, 1/0.225])

samples = denormalize(samples).clamp(0.0, 1.0)
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.tight_layout()
plt.show()

