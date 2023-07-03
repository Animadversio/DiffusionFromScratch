# Diffusion From Scratch

Binxu Wang (binxu_wang@hms.harvard.edu)

Tutorial on Stable Diffusion Models at ML from Scratch seminar series at Harvard.  
![](https://scholar.harvard.edu/sites/scholar.harvard.edu/files/styles/os_files_large/public/binxuw/files/diffusion_proc1.gif?m=1667441103&itok=y1BDYFl1)
* [Homepage](https://scholar.harvard.edu/binxuw/classes/machine-learning-scratch/materials/stable-diffusion-scratch)
* [Tutorial Slides](https://scholar.harvard.edu/files/binxuw/files/stable_diffusion_a_tutorial.pdf) 

This tiny self-contained code base allows you to 
* Rebuild the Stable Diffusion Model in a single Python script. 
* Train your toy version of stable diffusion on classic datasets like MNIST, CelebA
![](https://scholar.harvard.edu/sites/scholar.harvard.edu/files/styles/os_files_xxlarge/public/binxuw/files/stablediffusion_overview.jpg?m=1667438590&itok=n2gM0Xba)
## Colab notebooks
* Playing with Stable Diffusion and inspecting the internal architecture of the models. [Open in Colab](https://colab.research.google.com/drive/1TvOlX2_l4pCBOKjDI672JcMm4q68sKrA?usp=sharing)
* Build your own Stable Diffusion UNet model from scratch in a notebook. (with < 300 lines of codes!) [Open in Colab](https://colab.research.google.com/drive/1mm67_irYu3qU3hnfzqK5yQC38Fd5UFam?usp=sharing)
    * [Self contained script](https://github.com/Animadversio/DiffusionFromScratch/blob/master/StableDiff_UNet_model.py)
    * [Unit tests](https://github.com/Animadversio/DiffusionFromScratch/blob/master/StableDiff_UNet_unittest.py)
* Build a Diffusion model (with UNet + cross attention) and train it to generate MNIST images based on the "text prompt". [Open in Colab (exercise)](https://colab.research.google.com/drive/1Y5wr91g5jmpCDiX-RLfWL1eSBWoSuLqO?usp=sharing) [Open in Colab (answer)](https://colab.research.google.com/drive/1_MEFfBdOI06GAuANrs1b8L-BBLn3x-ZJ?usp=sharing)
