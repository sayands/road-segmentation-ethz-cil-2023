import torch
from PIL import Image
from transformers import SamModel, SamProcessor
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
import matplotlib.pyplot as plt
import gc


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color is None:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()


def show_masks_on_image(raw_images, batch_masks):
    for idx in range(len(raw_images)):
        raw_image = raw_images[idx]
        masks = batch_masks[idx]["masks"]
        np_image = np.array(raw_image)
        torch_image = torch.from_numpy(np_image)
        plt.imshow(np_image)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        for mask in masks:
            show_mask(mask,ax=ax, random_color=True)
            # masked_part = torch_image[mask]
            # mean_pixels = torch.mean(masked_part.float(), 0)
            # red, green, blue = mean_pixels[0]/255., mean_pixels[1]/255., mean_pixels[2]/255.
            # max_channel = max(red, green, blue)
            # min_channel = min(red, green, blue)
            # delta = max_channel - min_channel
            # L = (max_channel + min_channel) / 2
            # S = delta / (1 - abs(2 * L - 1))
            # if S < 0.1:
            #     show_mask(mask, ax=ax, random_color=False, color=np.array([1, 0, 0, .6]))
            # elif 0.1 < S < 0.3:
            #     show_mask(mask, ax=ax, random_color=False, color=np.array([0, 1, 0, .6]))
            # elif 0.3 < S < 0.5:
            #     show_mask(mask, ax=ax, random_color=False, color=np.array([0, 0, 1, .6]))
            # elif 0.5 < S < 0.7:
            #     show_mask(mask, ax=ax, random_color=False, color=np.array([1, 1, 0, .6]))
            # elif 0.7 < S < 0.9:
            #     show_mask(mask, ax=ax, random_color=False, color=np.array([0, 1, 1, .6]))
            # else:
            #     show_mask(mask, ax=ax, random_color=False, color=np.array([1, 0, 1, .6]))
        plt.axis("off")
        plt.show()
        del mask
        gc.collect()

class SAM():
    def __init__(self):
        self.generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)

    def predict(self, images_path):
        if type(images_path) is list:
            raw_imgs = [Image.open(img) for img in images_path]
        else:
            raw_imgs = [Image.open(images_path)]
        outputs = self.generator(raw_imgs, points_per_batch=256)
        show_masks_on_image(raw_imgs, outputs)


if __name__ == '__main__':
    sam = SAM()
    import os
    paths = os.listdir('/home/ivan/PycharmProjects/ETH/road-segmentation-ethz-cil-2023/data/3_ZOOM_18')
    images = []
    for path in paths:
        if 'label' not in path:
            images.append(f"/home/ivan/PycharmProjects/ETH/road-segmentation-ethz-cil-2023/data/3_ZOOM_18/{path}")
    sam.predict("/home/ivan/PycharmProjects/ETH/road-segmentation-ethz-cil-2023/data/3_ZOOM_18/1.png")
