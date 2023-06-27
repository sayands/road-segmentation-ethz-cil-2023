import argparse
import logging
import json
import os
import re

import numpy as np
import torch
import cv2
import PIL
from PIL import Image as im

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

import sys
sys.path.append('..')
from configs import config, update_config

foreground_threshold = 0.25 # percentage of pixels of val 255 required to assign a foreground label to a patch
img_number = -1

# assign a label to a patch
def patch_to_label(patch):
    patch = patch.astype(np.float64) / 255
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def mask_to_submission_strings(image_filename, mask_dir=None):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", os.path.basename(image_filename)).group(0))
    im = PIL.Image.open(image_filename)
    im_arr = np.asarray(im)
    if len(im_arr.shape) > 2:
        # Convert to grayscale.
        im = im.convert("L")
        im_arr = np.asarray(im)

    patch_size = 16
    mask = np.zeros_like(im_arr)
    for j in range(0, im_arr.shape[1], patch_size):
        for i in range(0, im_arr.shape[0], patch_size):
            patch = im_arr[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            mask[i:i+patch_size, j:j+patch_size] = int(label*255)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

#     if mask_dir:
#         save_mask_as_img(mask, os.path.join(mask_dir, "mask_" + image_filename.split("/")[-1]))
    

def save_mask_as_img(img_arr, mask_filename):
    img = PIL.Image.fromarray(img_arr)

    # Temporarily resizing image to 400x400 using nearest neighbor interpolation
    img = img.resize((400,400), im.NEAREST)

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    img.save(mask_filename)

def test(config):
    # Load model
    model = smp.DeepLabV3Plus(encoder_name='efficientnet-b3', encoder_depth=5, encoder_weights='imagenet', 
                              encoder_output_stride=16, decoder_channels=256, decoder_atrous_rates=(12, 24, 36), 
                              in_channels=3, classes=2, activation=None, upsampling=4, aux_params=None)
    
    model.load_state_dict((torch.load(config["test"]["model_path"]))['model'])

    # Load and evaluate images
    for path in os.listdir(config["test"]["test_path"]):
        im_path = os.path.join(os.path.abspath(config["test"]["test_path"]), path)
        # Set model to evaluation
        model.eval()

        # No gradient tracking
        with torch.no_grad():
            # Load image
            image = cv2.imread(im_path)
            image = image[:, :, [2, 1, 0]] # Convert to RGB
            image = image.astype(np.float32)
            image /= 255.0

            # Resize image
            image = cv2.resize(image, (256, 256))

            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image).to(config["test"]["device"])

            # make prediction
            prediction_mask = model(image)

            prediction_mask = prediction_mask[0].cpu().numpy()
            prediction_mask = np.transpose(prediction_mask, (1, 2, 0))
            prediction_mask = np.argmax(prediction_mask, axis=2)
            prediction_mask *= 255
            prediction_mask = prediction_mask.astype(np.uint8)

            pred_mask = np.reshape(prediction_mask, (256,256))
            save_mask_as_img(pred_mask, os.path.join(config["test"]["mask_results_path"], "mask_" + im_path.split("/")[-1]))

def masks_to_submission(submission_filename, mask_dir, *image_filenames):
    os.makedirs(os.path.dirname(submission_filename), exist_ok=True)
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, mask_dir=mask_dir))

def parse_args():
    parser = argparse.ArgumentParser(description="CIL Project")
    parser.add_argument(
        "--config",
        type=str,
        default="test.yaml",
        help="Path to config file to replace defaults",
    )
    args = parser.parse_args()
    return parser, args 

def main():
    parser, args = parse_args()
    cfg = update_config(config, args.config)

    # Runs the model over test set and saves the predicted mask images
    # test(cfg)

    # gets the predicted mask images and converts to submission
    image_filenames = [os.path.join(config["test"]["mask_results_path"], name) for name in os.listdir(config["test"]["mask_results_path"])]
    masks_to_submission(config["test"]["submission_path"], "", *image_filenames)

if __name__ == '__main__':
    main()