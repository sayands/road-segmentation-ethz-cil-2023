import logging as log
import os
import sys

import numpy as np
import torch
import wandb
from torch import nn
from tqdm import tqdm

sys.path.append('.')
from utils import common
from src.tools.losses import DiceLoss, FocalTverskyLoss
from src.tools.metrics import eval_metrics


class Trainer(nn.Module):
    def __init__(self, config, model, log_dir, device):
        super().__init__()
        self.device = device
        self.cfg = config
        self.model = model.to(self.device)
        self.log_dir = log_dir
        self.log_dict = {}

        self.log_phase = {"train" : self.log_dict.copy(), "valid": self.log_dict.copy()}
        
        self.init_loss()
        self.init_optimizer()
        self.init_log_dict()

    def init_optimizer(self):
        trainable_parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
        )
    
    def init_loss(self):
        self.loss_change = False if len(self.cfg["loss"]["epochs"]) == 1 else True
        self.loss_type = self.cfg["loss"]["loss_type"][0]
        self.loss_wlambda = self.cfg["loss"]["wlambda"][0]
        self.loss_alpha = self.cfg["loss"]["alpha"][0]
        self.loss_gamma = self.cfg["loss"]["gamma"][0]
        self.loss_currpointer = 0
        
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_dice = DiceLoss()
        self.loss_ftl = FocalTverskyLoss(alpha=self.loss_alpha,
                                             gamma=self.loss_gamma)


    def init_log_dict(self, phase="train"):
        """Custom log dict."""

        if phase == "train":
            self.log_dict = self.log_phase["train"]
        elif phase == "valid":
            self.log_dict = self.log_phase["valid"]

        self.log_dict["total_loss"] = 0.0
        self.log_dict["ce_loss"] = 0.0
        self.log_dict["dice_loss"] = 0.0
        self.log_dict["ftl_loss"] = 0.0

        self.log_dict["total_iter_count"] = 0
        self.log_dict["image_count"] = 0

        self.log_dict["accuracy_metric"] = 0.0
        self.log_dict["precision_metric"] = 0.0
        self.log_dict["recall_metric"] = 0.0
        self.log_dict["F1_metric"] = 0.0
        self.log_dict["mIoU_metric"] = 0.0

    def forward(self):
        """Forward pass of the network."""
        self.prediction = self.model(self.images)
    def compute_loss(self, seg_pred, seg_gt, epoch, phase="train"):
        """Compute loss """

        if self.loss_change and epoch > (self.cfg["loss"]["epochs"])[self.loss_currpointer]:
            self.loss_currpointer += 1
            self.loss_type = (self.cfg["loss"]["loss_type"])[self.loss_currpointer]
            self.loss_wlambda = (self.cfg["loss"]["wlambda"])[self.loss_currpointer]
            self.loss_alpha = (self.cfg["loss"]["alpha"])[self.loss_currpointer]
            self.loss_gamma = (self.cfg["loss"]["gamma"])[self.loss_currpointer]

        if phase=="train":
            self.log_dict = self.log_phase["train"]
        elif phase == "valid":
            self.log_dict = self.log_phase["valid"]
        if self.loss_type == 'ce':
            loss = self.loss_ce(seg_pred, torch.squeeze(torch.argmax(seg_gt, dim=1)))
            self.log_dict["ce_loss"] += loss.item()
        elif self.loss_type == 'dice':
            loss = self.loss_dice(seg_pred, seg_gt)
            self.log_dict["dice_loss"] += loss.item()
        elif self.loss_type == 'ftl':
            loss = self.loss_ftl(seg_pred, seg_gt, self.loss_alpha, self.loss_gamma)
            self.log_dict["ftl_loss"] += loss.item()
        elif self.loss_type == 'ce+dice':
            loss_ce = self.loss_ce(seg_pred, torch.squeeze(torch.argmax(seg_gt, dim=1)))
            self.log_dict["ce_loss"] += loss_ce.item()
            loss_dice = self.loss_dice(seg_pred, seg_gt)
            self.log_dict["dice_loss"] += loss_dice.item()
            loss = (loss_ce * self.loss_wlambda) + (loss_dice * (1-self.loss_wlambda))
        elif self.loss_type == 'ce+ftl':
            loss_ce = self.loss_ce(seg_pred, torch.squeeze(torch.argmax(seg_gt, dim=1)))
            self.log_dict["ce_loss"] += loss_ce.item()
            loss_ftl = self.loss_ftl(seg_pred, seg_gt, self.loss_alpha, self.loss_gamma)
            self.log_dict["ftl_loss"] += loss_ftl.item()
            loss = (loss_ce * self.loss_wlambda) + (loss_ftl * (1-self.loss_wlambda))
        return loss

    def backward(self):
        """Backward pass of the network."""
        # Compute loss
        loss = self.compute_loss(self.prediction, self.seg_gt, self.epoch)
        self.log_dict['total_loss'] += loss.item()
        metrics = eval_metrics(self.prediction, self.seg_gt)
        for k in self.log_dict.keys():
            if "metric" in k:
                self.log_dict[k] += metrics[k.replace("_metric", "")]
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

    def log(self, step, epoch, phase):
        """Log the training information."""
        self.log_dict = self.log_phase[phase]
        log_text = f"PHASE: {phase}, STEP {step} - EPOCH {epoch}/{self.cfg['train']['epochs']}"
        for k in self.log_dict.keys():
            if "loss" in k or "metric" in k:
                self.log_dict[k] /= self.log_dict["total_iter_count"]
                log_text += f" | {k}: {self.log_dict[k]:>.3E}"

        log.info(log_text)

        for key, value in self.log_dict.items():
            if "loss" in key or "metric" in key:
                wandb.log({f"{phase}_{key}": value}, step=step)
        self.init_log_dict(phase=phase)

    def validate(self, loader, img_shape, step=0, epoch=0, save_result=False):
        """validation function for generating final results."""
        torch.cuda.empty_cache()  # To avoid CUDA out of memory
        self.eval()
        self.init_log_dict(phase="valid")

        log.info("Beginning validation...")
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
                valid_loss = self.compute_loss(seg_pred, seg_gt, epoch, phase="valid")
                self.log_dict['total_loss'] += valid_loss
                metrics = eval_metrics(seg_pred, seg_gt)
                for k in self.log_dict.keys():
                    if "metric" in k:
                        self.log_dict[k] += metrics[k.replace("_metric", "")]

                # Save first in the batch
                # img_to_save = (images[0]).reshape(*img_shape, 3).cpu().numpy() * 255
                img_to_save = images[0].cpu().numpy() * 255
                img_to_save = np.transpose(img_to_save, (1, 2, 0))

                seg_gt_to_save = seg_gt[0].cpu().numpy()
                seg_gt_to_save = np.transpose(seg_gt_to_save, (1, 2, 0))
                seg_gt_to_save = (1 - seg_gt_to_save[..., 0]) * seg_gt_to_save[..., 1]
                seg_gt_to_save *= 255

                seg_pred_to_save = seg_pred[0].cpu().numpy()
                seg_pred_to_save = np.transpose(seg_pred_to_save, (1, 2, 0))
                seg_pred_to_save = np.argmax(seg_pred_to_save, axis=2)
                seg_pred_to_save *= 255

                wandb_img.append(wandb.Image(img_to_save))
                wandb_seg_gt.append(wandb.Image(seg_gt_to_save))
                wandb_seg_pred.append(wandb.Image(seg_pred_to_save))

        wandb.log({"Ground-truth Labels": wandb_seg_gt}, step=step)
        wandb.log({"Predicted Labels": wandb_seg_pred}, step=step)
        wandb.log({"Images": wandb_img}, step=step)

        self.log_dict["total_iter_count"] = len(loader)

        # self.log_dict['total_loss'] /= len(loader)
        # log_text = f"EPOCH {epoch}/{self.cfg['train']['epochs']}"
        # log_text += f"{'Loss'} | {valid_loss:.2f}"
        # wandb.log({"Loss": valid_loss, "Epoch": epoch}, step=step)
        # log.info(log_text)

        self.train()

    def save_model(self, epoch, dirname):
        """Save the model checkpoint."""
        dirpath = os.path.join(self.log_dir, dirname)
        common.ensure_dir(dirpath)

        model_state_dict = self.model.state_dict()
        fname = os.path.join(dirpath, f'{epoch}.pth')
        log.info(f"Saving model checkpoint to: {fname}")

        state_dict = {
            'optimiser': self.optimizer,
            'model': model_state_dict,
            'epoch': self.epoch
        }

        torch.save(state_dict, fname)
