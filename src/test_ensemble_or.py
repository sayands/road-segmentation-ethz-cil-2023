import argparse
import os
import re
import sys
import tqdm

import PIL
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt

sys.path.append('..')
from configs import config, update_config

foreground_threshold = 0.25  # percentage of pixels of val 255 required to assign a foreground label to a patch
img_number = -1

WINDOW_SIZE = 256


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
            mask[i:i + patch_size, j:j + patch_size] = int(label * 255)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


#     if mask_dir:
#         save_mask_as_img(mask, os.path.join(mask_dir, "mask_" + image_filename.split("/")[-1]))


def save_mask_as_img(img_arr, mask_filename):
    img = PIL.Image.fromarray(img_arr)

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    img.save(mask_filename)


def pad_image(image, start_widths, auto_pad, padding):
    height, width, rgb = image.shape
    if auto_pad is False:
        padded_width = width + 2 * padding
        padded_height = height + 2 * padding
    else:
        padding = (start_widths[-1] + WINDOW_SIZE - width) // 2

        padded_width = width + 2 * padding
        padded_height = height + 2 * padding

    padded_image = np.zeros((padded_width, padded_height, rgb)).astype(np.float32)
    padded_image[padding:padding+height, padding:padding+width] = image
    return padding, padded_image


def generate_starting_points(max_value, stride):
    points = [0]
    while points[-1] + WINDOW_SIZE < max_value:
        points.append(points[-1] + stride)
    return points


def generate_crops(image, start_widths, start_heights):
    height, width, _ = image.shape

    for start_width in start_widths:
        for start_height in start_heights:
            yield start_height, start_width, image[start_width:start_width + WINDOW_SIZE,
                                                   start_height:start_height + WINDOW_SIZE, :]


def avrg_mask(full_size_mask, stride, height, width):

    avrg_matrix = np.zeros((full_size_mask.shape[1], full_size_mask.shape[2], 1))

    start_heights = generate_starting_points(height, stride)
    start_widths = generate_starting_points(width, stride)
    for height in start_heights:
        for width in start_widths:
            avrg_matrix[height:height + WINDOW_SIZE, width:width + WINDOW_SIZE] += 1
    return full_size_mask / avrg_matrix


def get_mask(full_mask, stride, padding, original_image_x, original_image_y):
    prediction_mask = avrg_mask(full_mask, stride, original_image_x, original_image_y)
    
    prediction_mask = np.argmax(prediction_mask, axis=-1)
    prediction_mask = np.sum(prediction_mask, axis=0)
    prediction_mask[prediction_mask>0] = 1

    prediction_mask *= 255
    prediction_mask = prediction_mask.astype(np.uint8)
    # remove the padding
    return prediction_mask[padding:padding+original_image_x, padding:padding+original_image_y]


def test(config):

    # Load model
    model_ensemble_name = config["test"]["model_ensemble_name"]
    model_ensemble_path = config["test"]["model_ensemble_path"]
    model_ensemble = []
    for i in range(len(model_ensemble_name)):
        model = smp.DeepLabV3Plus(encoder_name=model_ensemble_name[i], encoder_depth=5, encoder_weights='imagenet',
                              encoder_output_stride=16, decoder_channels=256, decoder_atrous_rates=(12, 24, 36),
                              in_channels=3, classes=2, activation=None, upsampling=4, aux_params=None).to(
            config["test"]["device"])
        model.load_state_dict((torch.load(model_ensemble_path[i]))['model'])
        model.eval()
        model_ensemble.append(model)

    # Load and evaluate images
    for path in tqdm.tqdm(os.listdir(config["test"]["test_path"])):
        im_path = os.path.join(os.path.abspath(config["test"]["test_path"]), path)
        # Set model to evaluation

        # No gradient tracking
        with torch.no_grad():
            # Load image
            original_image = cv2.imread(im_path)
            original_image = original_image[:, :, [2, 1, 0]]  # Convert to RGB
            original_image = original_image.astype(np.float32)
            original_image /= 255.0

            stride = config["test"]["stride"]

            start_heights = generate_starting_points(original_image.shape[0], stride)
            start_widths = generate_starting_points(original_image.shape[1], stride)

            padding, padded_image = pad_image(original_image, start_widths, config["test"]["auto_padding"],
                                     config["test"]["padding"])
            full_size_mask = np.zeros((len(model_ensemble_name), *padded_image.shape[:2], 2))

            for start_height, start_width, image in generate_crops(padded_image, start_widths, start_heights):
                image = np.transpose(image, (2, 0, 1))
                image = np.expand_dims(image, 0)
                image = torch.from_numpy(image).to(config["test"]["device"])

                # make prediction
                prediction_mask = [np.transpose(model(image)[0].cpu().numpy(), (1, 2, 0)) for model in model_ensemble]
                # prediction_mask = prediction_mask[0].cpu().numpy()
                # prediction_mask = np.transpose(prediction_mask, (1, 2, 0))
                
                for i in range(len(model_ensemble_name)):
                    full_size_mask[i, start_width: start_width + WINDOW_SIZE, start_height:start_height + WINDOW_SIZE,
                                   :] += prediction_mask

            prediction_mask = get_mask(full_size_mask, stride, padding, original_image.shape[0], original_image.shape[1])
            save_mask_as_img(prediction_mask,
                             os.path.join(config["test"]["mask_results_path"], "mask_" + im_path.split("/")[-1]))


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
        default="../configs/base_test_ensemble.yaml",
        help="Path to config file to replace defaults",
    )
    args = parser.parse_args()
    return parser, args


def main():
    parser, args = parse_args()
    cfg = update_config(config, args.config)

    # Runs the model over test set and saves the predicted mask images
    test(cfg)

    # gets the predicted mask images and converts to submission
    image_filenames = [os.path.join(config["test"]["mask_results_path"], name) for name in
                       os.listdir(config["test"]["mask_results_path"])]
    masks_to_submission(config["test"]["submission_path"], "", *image_filenames)


if __name__ == '__main__':
    main()
