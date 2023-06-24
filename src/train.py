import argparse
import random
import sys
import os
import logging
import json
import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import wandb

import sys
sys.path.append('..')
from utils import common
from configs import config, update_config

from src.datasets.aerial_data import AerialSeg
from src.tools.trainer import Trainer

def train(config):
    # Set random seed.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    log_dir = config.log_dir
    common.ensure_dir(log_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialise dataset and dataloader
    train_dataset = AerialSeg(config, split='training')
    valid_dataset = AerialSeg(config, split='validation')
    
    img_shape = train_dataset.img_shape
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config["validation"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Initialise network and trainer: efficientnet-b3, resnet34
    model = smp.DeepLabV3Plus(encoder_name='efficientnet-b3', encoder_depth=5, encoder_weights='imagenet', 
                              encoder_output_stride=16, decoder_channels=256, decoder_atrous_rates=(12, 24, 36), 
                              in_channels=3, classes=2, activation=None, upsampling=4, aux_params=None)
    
    trainer = Trainer(config, model, log_dir, device)
    # Setup wandb for logging
    if config["wandb"]["id"] is not None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        wandb_id = f'{config["wandb"]["id"]}_{timestamp}' 

    else:
        wandb_id = wandb.util.generate_id()
        with open(os.path.join(log_dir, "wandb_id.txt"), "w+", encoding="UTF-8") as f:
            f.write(wandb_id)

    wandb_mode = "disabled" if (not config["wandb"]["wandb"]) else "online"
    wandb.init(
        id=wandb_id,
        entity=config["wandb"]["entity"],
        project=config["wandb"]["project"],
        name=config["wandb"]["name"],
        group=config["wandb"]["group"],
        resume="allow",
        config=config,
        settings=wandb.Settings(start_method="fork"),
        mode=wandb_mode,
        dir=log_dir,
    )
    wandb.watch(model)
    
    # Main training loop
    global_step = 0
    for epoch in range(1, config["train"]["epochs"] + 1):
        for data in train_loader:
            trainer.step(data, epoch)
            if global_step % config["train"]["log_every"] == 0:
                trainer.log(global_step, epoch, phase="train")
            global_step += 1

        if epoch % config["train"]["save_every"] == 0:
            trainer.save_model(epoch, wandb_id)

        if epoch % config["validation"]["valid_every"] == 0:
            trainer.validate(
                valid_loader,
                img_shape,
                step=global_step,
                epoch=epoch)
            trainer.log(global_step, epoch, phase="valid")

    wandb.finish()
    

def parse_args():
    parser = argparse.ArgumentParser(description="CIL Project")
    parser.add_argument(
        "--config",
        type=str,
        default="train.yaml",
        help="Path to config file to replace defaults",
    )
    args = parser.parse_args()
    return parser, args   
    

def main():
    parser, args = parse_args()
    cfg = update_config(config, args.config)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=config["log_level"],
        format="%(asctime)s|%(levelname)8s| %(message)s",
        handlers=handlers,
    )
    message = json.dumps(cfg, indent=4)
    logging.info(f"Info: \n{message}")
    
    train(cfg)

if __name__ == '__main__':
    main()