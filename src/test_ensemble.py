import argparse
import os
import re
import sys

import PIL
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import tqdm

sys.path.append('..')
from configs import config, update_config

foreground_threshold = 0.25  # percentage of pixels of val 255 required to assign a foreground label to a patch
img_number = -1

WINDOW_SIZE = 256


# assign a label to a patch
def patch_to_label(patch):
    """
    @param patch: np.array of the portion which has to be patched
    @return: 1 if mean passes the threshold otherwise 0
    """
    patch = patch.astype(np.float64) / 255
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """
    Reads a single image and outputs the strings that should go into the submission file
    @param image_filename: image to be parsed to submission format
    @rtype: the submission string in csv row format
    """
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


def calculate_metrics(gt_mask, pred_mask):
    """
    Calculate True Positives (TP), False Positives (FP),
    True Negatives (TN), and False Negatives (FN) for binary segmentation task.

    @param gt_mask (numpy.ndarray): Ground truth binary mask (0 or 1).
    @param pred_mask (numpy.ndarray): Predicted binary mask (0 or 1).

    @rtype:int: True Positives (TP)
        int: False Positives (FP)
        int: True Negatives (TN)
        int: False Negatives (FN)
    """
    # Ensure the input masks have the same shape
    gt_mask = gt_mask[:, :, 0]
    assert gt_mask.shape == pred_mask.shape, "Ground truth and predicted masks must have the same shape."

    gt_mask = (gt_mask / 255.).astype(np.uint8)
    pred_mask = (pred_mask / 255.).astype(np.uint8)
    # Flatten the masks to 1D arrays
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()

    # Calculate True Positives (TP), False Positives (FP),
    # True Negatives (TN), and False Negatives (FN)
    tp = np.sum(np.logical_and(gt_flat == 1, pred_flat == 1))
    fp = np.sum(np.logical_and(gt_flat == 0, pred_flat == 1))
    tn = np.sum(np.logical_and(gt_flat == 0, pred_flat == 0))
    fn = np.sum(np.logical_and(gt_flat == 1, pred_flat == 0))

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    return precision, recall, f1_score


def save_mask_as_img(img_arr, mask_filename):
    """
    @param img_arr: the image as np.array
    @param mask_filename: location where the file is saved
    """
    img = PIL.Image.fromarray(img_arr)

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    img.save(mask_filename)


def pad_image(image, start_widths, auto_pad, padding):
    """
    @param image: image as np.array
    @param start_widths: the pre-computed start points of the sliding window
    @param auto_pad: if we should apply auto-padding - which will result in the minimal needed padding
    @param padding: pre-defined padding samples
    @return: padding size, padding image
    """
    height, width, rgb = image.shape
    if auto_pad is False:
        padded_width = width + 2 * padding
        padded_height = height + 2 * padding
    else:
        padding = (start_widths[-1] + WINDOW_SIZE - width) // 2

        padded_width = width + 2 * padding
        padded_height = height + 2 * padding

    padded_image = np.zeros((padded_width, padded_height, rgb)).astype(np.float32)
    padded_image[padding:padding + height, padding:padding + width] = image
    return padding, padded_image


def generate_starting_points(start_value, max_value, window_size, stride):
    """
    Generates and return a set of starting points of the windows.
    @param start_value: starting value from which the slide should start
    @param max_value: the maximum index available (size of image)
    @param window_size: the size of the window (both Height and Width)
    @param stride: stride or step for sliding
    @return: starting values
    """
    start_values = [w for w in range(start_value, max_value - window_size, stride)]
    # if the stride skips on the very last bit of the image we should still look into it.
    if max_value - window_size not in start_values:
        start_values.append(max_value - window_size)
    return start_values


def generate_crops(image, stride):
    """
    Generates crops of size WINDOW_SIZE x WINDOW_SIZE with steps equal to the given stride.
    @param image: the image which will be cropped
    @param stride: the step size for the window slide
    """
    height, width, _ = image.shape
    start_heights = generate_starting_points(0, height, WINDOW_SIZE, stride)
    start_widths = generate_starting_points(0, width, WINDOW_SIZE, stride)
    if width - WINDOW_SIZE not in start_widths:
        start_widths.append(width - WINDOW_SIZE)
    for start_width in start_widths:
        for start_height in start_heights:
            yield start_height, start_width, image[start_width:start_width + WINDOW_SIZE,
                                             start_height:start_height + WINDOW_SIZE, :]


def avrg_mask(full_size_mask, stride):
    """
    Function which averages full_size_mask and return the proper prediction mask
    @param full_size_mask: np.array matrix with all the predictions stacked onto each other
    @param stride: the stride with which the predictions have been done
    @return: the averaged out mask
    """
    height, width, _ = full_size_mask.shape

    avrg_matrix = np.zeros((height, width, 1))

    start_heights = generate_starting_points(0, height, WINDOW_SIZE, stride)
    start_widths = generate_starting_points(0, width, WINDOW_SIZE, stride)
    for height in start_heights:
        for width in start_widths:
            avrg_matrix[height:height + WINDOW_SIZE, width:width + WINDOW_SIZE] += 1
    return full_size_mask / avrg_matrix


def get_mask(full_mask, stride, padding, original_image_x, original_image_y):
    """
    @param full_mask: full prediction mask (including paddings)
    @param stride: the stride with which image was cropped
    @param padding: the padding size of the image
    @param original_image_x: image width
    @param original_image_y: image height
    @return: the final prediction mask for the image corresponding to the full_mask
    """
    prediction_mask = avrg_mask(full_mask, stride)
    prediction_mask = np.argmax(prediction_mask, axis=2)
    prediction_mask *= 255
    prediction_mask = prediction_mask.astype(np.uint8)
    # remove the padding
    return prediction_mask[padding:padding + original_image_x, padding:padding + original_image_y]


def get_paths(validation_set_file='/home/ivan/PycharmProjects/ETH/road-segmentation-ethz-cil-2023/data/seg-data/validation.txt',
              number_of_validation_files=200):
    """
    Returns a list of files to use for validation.
    @param validation_set_file: file which contains the validation images
    @param number_of_validation_files: how many files should be returned
    @return: filenames listed in the file
    """
    with open(validation_set_file, 'r') as f:
        lines = f.readlines()
        lines = [line[:-1] + '.png' for line in lines]
        return lines[:number_of_validation_files]


def init_model(architecture_name):
    """
    Initialize model according to the given architecture name
    @param architecture_name: the name of the architecture
    @return: initialized model with the architecture_name as encoder
    """
    try:
        return smp.UnetPlusPlus(encoder_name=architecture_name, encoder_depth=5, encoder_weights='imagenet',
                                decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16),
                                decoder_attention_type=None, in_channels=3, classes=2, activation=None,
                                aux_params=None)
    except RuntimeError:
        return smp.DeepLabV3Plus(encoder_name=architecture_name, encoder_depth=5, encoder_weights='imagenet',
                                 encoder_output_stride=16, decoder_channels=256, decoder_atrous_rates=(12, 24, 36),
                                 in_channels=3, classes=2, activation=None, upsampling=4, aux_params=None)


def test(config):
    """
    Equivalent to main function which will run and test model's defined in the config file passed
    @param config: Configuration for the model load and testing
    """
    # Load model
    model_ensemble_name = config["test"]["model_ensemble_name"]
    model_ensemble_path = config["test"]["model_ensemble_path"]
    model_ensemble = []
    for i in range(len(model_ensemble_name)):
        model = init_model(model_ensemble_name[i]).to(config["test"]["device"])
        model.load_state_dict((torch.load(model_ensemble_path[i]))['model'])
        model.eval()
        model_ensemble.append(model)

    precisions = []
    recalls = []
    f1_scores = []
    # Load and evaluate images
    paths = get_paths()
    # Iterate over validation paths
    for path in tqdm.tqdm(paths):
        im_path = os.path.join(os.path.abspath(config["test"]["test_path"]), path)
        # Set model to evaluation

        gt_path = None
        if config["test"]["test_groundtruth_path"] != '':
            gt_path = os.path.join(os.path.abspath(config["test"]["test_groundtruth_path"]), path)

        # No gradient tracking
        with torch.no_grad():
            # Load image
            full_size_image = cv2.imread(im_path)
            full_size_image = full_size_image[:, :, [2, 1, 0]]  # Convert to RGB
            full_size_image = full_size_image.astype(np.float32)
            full_size_image /= 255.0

            full_size_mask = np.zeros((*full_size_image.shape[:2], 2))
            for start_height, start_width, image in generate_crops(full_size_image, config["test"]["stride"]):
                image = np.transpose(image, (2, 0, 1))
                image = np.expand_dims(image, 0)
                image = torch.from_numpy(image).to(config["test"]["device"])

                # make prediction
                prediction_mask = model_ensemble[0](image)
                for i in range(1, len(model_ensemble)):
                    prediction_mask += model_ensemble[i](image)
                prediction_mask /= len(model_ensemble)

                prediction_mask = prediction_mask[0].cpu().numpy()
                prediction_mask = np.transpose(prediction_mask, (1, 2, 0))
                full_size_mask[start_width: start_width + WINDOW_SIZE, start_height:start_height + WINDOW_SIZE, :] \
                    += prediction_mask

            # Get final prediction
            prediction_mask = get_mask(full_size_mask, 256, 0, full_size_image.shape[0], full_size_image.shape[1])

            if gt_path is not None: gt_mask = cv2.imread(gt_path)
            precision, recall, f1_score = calculate_metrics(gt_mask, prediction_mask)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

            save_mask_as_img(prediction_mask,
                             os.path.join(config["test"]["mask_results_path"], "mask_" + im_path.split("/")[-1]))

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)

    print(f'{model_ensemble_name} & {np.mean(precisions)} & { np.mean(recalls)} & {np.mean(f1_scores)}')


def masks_to_submission(submission_filename, mask_dir, *image_filenames):
    """
    Transform the image predictions into submission ready format.
    @param submission_filename: the filename where the submission results will be saved
    @param mask_dir: directory where the prediction masks are stored
    @param image_filenames: the filenames of the masks
    """
    os.makedirs(os.path.dirname(submission_filename), exist_ok=True)
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, mask_dir=mask_dir))


def parse_args():
    """
    Loads and returns the configuration from a YAML file.
    """
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
    """
    Load configuration and run test function which will evaluate the perfomance of the model/models listed
    in the configuration
    """
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
