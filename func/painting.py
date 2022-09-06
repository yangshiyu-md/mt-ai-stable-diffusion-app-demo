from io import BytesIO
import os
from pickle import FALSE
import sys
from tkinter.tix import IMAGE
from torch import autocast
from torchvision.utils import make_grid
from einops import rearrange
import requests
import PIL
import torch
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2
import gc
from utils import paint_pipeline, image2image_pipeline, device

class GaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"
    
    def __init__(self, radius=100, bounds=None):
        self.radius = radius
        self.bounds = bounds
    
    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def inpaint_predict(dict, prompt, seed, num_images=4):
    gc.collect()
    torch.cuda.empty_cache()
    fixed_size = 512
    random.seed(seed)
    init_img = dict['image'].convert("RGB").resize((fixed_size, fixed_size))
    mask_img = dict['mask'].convert("RGB").resize((fixed_size, fixed_size))
    result_images = []
    with autocast("cuda"):
        for _ in range(num_images):
            images = paint_pipeline(prompt=prompt, init_image=init_img, mask_image=mask_img, strength=0.75)["sample"]
            result_images.append(images[0])
    
    torch_images = [rearrange(torch.tensor(np.array(pil_image)), 'h w c -> c h w') for pil_image in result_images]
    grids = torch.stack(torch_images, 0).to(device)
    n_rows = int(num_images ** 0.5)
    grids = make_grid(grids, n_rows, 0)
    grids = rearrange(grids, 'c h w -> h w c').cpu().numpy()
    grids = grids.astype(np.uint8)

    return grids, result_images


def outpaint_predict(image, prompt, seed, num_images=4):
    gc.collect()
    torch.cuda.empty_cache()
    fixed_size = 512
    expand_lenth = 32
    expanded_size = fixed_size + 2 * expand_lenth
    random.seed(seed)
    # create new expanded image
    init_img = image.convert("RGB").resize((fixed_size, fixed_size))
    init_img = np.array(init_img)
    init_img = cv2.copyMakeBorder(init_img, 
                                  expand_lenth, 
                                  expand_lenth, 
                                  expand_lenth, 
                                  expand_lenth, 
                                  cv2.BORDER_DEFAULT)
    init_img = Image.fromarray(init_img)
    # create mask
    left_upper = (expand_lenth, expand_lenth)
    mask_img = Image.new(mode="RGB", size=(expanded_size, expanded_size), color="white")
    invert_mask = Image.new(mode="RGB", size=(fixed_size, fixed_size), color="black")
    mask_img.paste(im=invert_mask, box=left_upper)
    mask_img = mask_img.filter(GaussianBlur(radius=5))
    
    result_images = []
    with autocast("cuda"):
        for _ in range(num_images):
            images = paint_pipeline(prompt=prompt, init_image=init_img, mask_image=mask_img, strength=0.75)["sample"]
            result_images.append(images[0])
    
    torch_images = [rearrange(torch.tensor(np.array(pil_image)), 'h w c -> c h w') for pil_image in result_images]
    grids = torch.stack(torch_images, 0).to(device)
    n_rows = int(num_images ** 0.5)
    grids = make_grid(grids, n_rows, 0)
    grids = rearrange(grids, 'c h w -> h w c').cpu().numpy()
    grids = grids.astype(np.uint8)

    return grids, result_images

def sketch_predict(init_img, prompt, seed, num_images=4):
    gc.collect()
    torch.cuda.empty_cache()
    fixed_size = 512
    random.seed(seed)
    init_img = init_img.convert("RGB").resize((fixed_size, fixed_size))
    result_images = []
    with autocast("cuda"):
        for _ in range(num_images):
            images = image2image_pipeline(prompt=prompt, init_image=init_img, strength=0.8)["sample"]
            result_images.append(images[0])
    
    torch_images = [rearrange(torch.tensor(np.array(pil_image)), 'h w c -> c h w') for pil_image in result_images]
    grids = torch.stack(torch_images, 0).to(device)
    n_rows = int(num_images ** 0.5)
    grids = make_grid(grids, n_rows, 0)
    grids = rearrange(grids, 'c h w -> h w c').cpu().numpy()
    grids = grids.astype(np.uint8)

    return grids, result_images