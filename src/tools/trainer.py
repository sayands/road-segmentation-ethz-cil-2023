import logging as log
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import wandb
from tqdm import tqdm
from PIL import Image
import segmentation_models_pytorch as smp

class Trainer(nn.Module):
    def __init__(self, config, model, log_dir, device):
        super().__init__()
        self.device = device
        self.cfg = config
        self.model = model.to(self.device)
        self.log_dir = log_dir
        self.log_dict = {}
        
        self.loss = smp.losses.JaccardLoss(mode='binary').to(device) # TODO : Can be modified
        
        self.init_optimizer()
        self.init_log_dict()
    
    def init_optimizer(self):
        trainable_parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
        )

    def init_log_dict(self):
        """Custom log dict."""
        self.log_dict["total_loss"] = 0.0
        self.log_dict["total_iter_count"] = 0
        self.log_dict["image_count"] = 0
    
    def forward(self):
        """Forward pass of the network."""
        self.prediction = self.model(self.images)
    
    def compute_loss(self, seg_pred, seg_gt):
        """Compute loss """
        loss = self.loss(seg_pred, seg_gt)
        return loss
    
    def backward(self):
        """Backward pass of the network."""
        # Compute loss
        loss = self.compute_loss(self.prediction, self.seg_gt)
        self.log_dict['total_loss'] += loss.item()
        loss.backward()
        
    def step(self, data, epoch):
        """A single training step."""
        self.epoch = epoch
        self.images = data["images"].to(self.device)
        self.seg_gt = data["groundtruth"].to(self.device)
        self.optimizer.zero_grad()
        
        self.forward()
        self.backward()

        self.optimizer.step()
        self.log_dict["total_iter_count"] += 1
        self.log_dict["image_count"] += self.images.shape[0]
    
    def log(self, step, epoch):
        """Log the training information."""
        log_text = f"STEP {step} - EPOCH {epoch}/{self.cfg['train']['epochs']}"
        self.log_dict["total_loss"] /= self.log_dict["total_iter_count"]
        log_text += f" | total loss: {self.log_dict['total_loss']:>.3E}"
        log.info(log_text)

        for key, value in self.log_dict.items():
            if "loss" in key:
                wandb.log({key: value}, step=step)
        self.init_log_dict()
    
    def validate(self, loader, img_shape, step=0, epoch=0, save_result=False):
        """validation function for generating final results."""
        torch.cuda.empty_cache()  # To avoid CUDA out of memory
        self.eval()
        
        log.info("Beginning validation...")
        valid_loss = 0.0
        if save_result:
            self.valid_img_dir = os.path.join(self.log_dir, "img")
            log.info(f"Saving segmentation result to {self.valid_img_dir}")
            if not os.path.exists(self.valid_img_dir):
                os.makedirs(self.valid_img_dir)
        
        wandb_img = []
        wandb_seg_gt = []
        wandb_seg_pred = []
        
        with torch.no_grad():
            for idx, data in enumerate(tqdm(loader)):
                images = data["images"].to(self.device)
                seg_gt = data["groundtruth"].to(self.device)
                
                seg_pred = self.model(images)
                valid_loss += self.compute_loss(seg_pred, seg_gt)
                
                
                # Save first in the batch
                img_to_save = (images[0]).reshape(*img_shape, 3).cpu().numpy() * 255
                
                seg_gt_to_save = seg_gt[0].cpu().numpy()
                seg_gt_to_save = np.transpose(seg_gt_to_save, (1, 2, 0))
                seg_gt_to_save = (1 - seg_gt_to_save[..., 0]) * seg_gt_to_save[..., 1] 
                seg_gt_to_save *= 255
                
                # TODO - @Ankita - save the image to wandb as you process (this should ideally be 2-channel output of scores) - current code faulty
                seg_pred_to_save = seg_pred[0].cpu().numpy()
                seg_pred_to_save = np.transpose(seg_pred_to_save, (1, 2, 0))
                seg_pred_to_save = np.argmax(seg_pred_to_save, axis=2)
                seg_pred_to_save *= 255
                
                wandb_img.append(wandb.Image(img_to_save))
                wandb_seg_gt.append(wandb.Image(seg_gt_to_save))
                wandb_seg_pred.append(wandb.Image(seg_pred_to_save))
        
        wandb.log({"Images": wandb_img}, step=step)
        wandb.log({"Ground-truth Labels": wandb_seg_gt}, step=step)
        wandb.log({"Predicted Labels": wandb_seg_pred}, step=step)
        
        valid_loss /= len(loader)
        log_text = f"EPOCH {epoch}/{self.cfg['train']['epochs']}"
        log_text += f"{'Loss'} | {valid_loss:.2f}"

        wandb.log({"Loss": valid_loss, "Epoch": epoch}, step=step)
        log.info(log_text)
        self.train()
    
    def save_model(self, epoch):
        """Save the model checkpoint."""
        model_state_dict = self.model.state_dict()
        fname = os.path.join(self.log_dir, f'{epoch}.pth')
        log.info(f"Saving model checkpoint to: {fname}")
        
        state_dict = {
            'optimiser' : self.optimizer,
            'model' : model_state_dict,
            'epoch' : self.epoch
        }
        
        torch.save(state_dict, fname)
                
                
            

                
                
                
                
        
