import argparse
import logging
import json
import os

import numpy as np
import torch
import cv2

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


def mask_to_submission_strings(im_arr, mask_dir=None):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number += 1
    if len(im_arr.shape) > 2:
        # Convert to grayscale.
        im_arr = im_arr.convert("L")
        im_arr = np.asarray(im_arr)

    patch_size = 16
    mask = np.zeros_like(im_arr)
    for j in range(0, im_arr.shape[1], patch_size):
        for i in range(0, im_arr.shape[0], patch_size):
            patch = im_arr[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            mask[i:i+patch_size, j:j+patch_size] = int(label*255)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

    # if mask_dir:
    #     save_mask_as_img(mask, os.path.join(mask_dir, "mask_" + image_filename.split("/")[-1]))

# def save_mask_as_img(img_arr, mask_filename):
#     img = PIL.Image.fromarray(img_arr)
#     os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
#     img.save(mask_filename)

def masks_to_submission(submission_filename, mask_dir, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, mask_dir=mask_dir))

def test(config):
    # Load model
    model = smp.DeepLabV3Plus(encoder_name='efficientnet-b3', encoder_depth=5, encoder_weights='imagenet', 
                              encoder_output_stride=16, decoder_channels=256, decoder_atrous_rates=(12, 24, 36), 
                              in_channels=3, classes=2, activation=None, upsampling=4, aux_params=None)
    
    model = model.load_state_dict(torch.load(config["test"]["model_path"]))

    os.makedirs(os.path.dirname(config["test"]["submission_path"]), exist_ok=True)
    with open(config["test"]["submission_path"], 'w') as f:
        f.write('id, prediction\n')

        # Load and evaluate images
        for path in os.listdir(config["test"]["test_path"]):
            # Set model to evaluation
            model.eval()

            # No gradient tracking
            with torch.no_grad():
                # Load image
                image = cv2.imread(path)
                image = image[:, :, [2, 1, 0]] # Convert to RGB
                image = image.astype(np.float32)
                image /= 255.0

                # Resize image
                image = cv2.resize(image, (128, 128))

                image = np.transpose(image, (2, 0, 1))
                image = np.expand_dims(image, 0)
                image = torch.from_numpy(image).to(config["test"]["device"])

                # make prediction
                prediction_mask = model(image).squeeze()
                prediction_mask = torch.sigmoid(prediction_mask)
                prediction_mask = prediction_mask.cpu().numpy()

                # Convert mask to submission
                f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(prediction_mask))

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
    test(cfg)

if __name__ == '__main__':
    main()