import torch
from torch.nn import functional as F
import lightning.pytorch as pl
import hydra
from jigsaw_latent.model.diffusion import DiffModel
from diffusers import DDPMScheduler
from tqdm import tqdm
from chamferdist import ChamferDistance
from jigsaw_latent.evaluation.jigsaw_evaluator import (
    calc_part_acc,
    rot_metrics,
    trans_metrics,
    randn_tensor
)
import numpy as np
from jigsaw_latent.model.modules.pn2 import PN2
from pytorch3d import transforms


class Jigsaw3D(pl.LightningModule):
    def __init__(self, cfg):
        super(Jigsaw3D, self).__init__()
        self.cfg = cfg
        self.diffusion = DiffModel(cfg)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.model.DDPM_TRAIN_STEPS,
            beta_schedule=cfg.model.DDPM_BETA_SCHEDULE,
            prediction_type=cfg.model.PREDICT_TYPE,
            beta_start=cfg.model.BETA_START,
            beta_end=cfg.model.BETA_END,
            clip_sample=False,
        )

        self.num_points = 27
        self.num_channels = 4

        self.noise_scheduler.set_timesteps(num_inference_steps=50)

        self.acc_list = []
        self.rmse_r_list = []
        self.rmse_t_list = []

        self.metric = ChamferDistance()
        self.encoder = PN2()


    def _apply_rots(self, part_pcs, noise_params):
        """
        Apply Noisy rotations to all points
        """
        noise_quat = noise_params[..., 3:]
        noise_quat = noise_quat / noise_quat.norm(dim=-1, keepdim=True)
        part_pcs = transforms.quaternion_apply(noise_quat.unsqueeze(2), part_pcs)
        
        return part_pcs


    def _apply_trans(self, xyz, noise_params):
        """
        Apply Noisy translation to the fps xyz
        """
        xyz = xyz + noise_params[..., :3].unsqueeze(2)
        return xyz
    

    def forward(self, data_dict):
        gt_trans = data_dict['part_trans']
        gt_rots = data_dict['part_rots']
        gt_rots_trans = torch.cat([gt_trans, gt_rots], dim=-1)

        noise = torch.randn(gt_rots_trans.shape, device=self.device)

        B, P, N, C = data_dict["part_pcs"].shape

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,),
                                  device=self.device).long()
        
        noisy_trans = self.noise_scheduler.add_noise(gt_rots_trans, noise, timesteps)

        # apply rotation and translation to part pcs
        part_pcs = self._apply_rots(data_dict['part_pcs'], noisy_trans)
        _, latent, xyz = self.encoder(part_pcs.flatten(0, 1).transpose(1, 2))

        xyz = xyz.reshape(B, P, self.num_points, 3)
        latent = latent.reshape(B, P, self.num_points, self.num_channels)

        # apply translation to fps xyz
        xyz = self._apply_trans(xyz, noisy_trans)

        pred_noise = self.diffusion(noisy_trans, timesteps, latent, 
                                    xyz, data_dict['part_valids'])

        output_dict = {
            'pred_noise': pred_noise,
            'gt_noise': noise
        }

        return output_dict


    def _loss(self, data_dict, output_dict):
        pred_noise = output_dict['pred_noise']
        part_valids = data_dict['part_valids'].bool()
        noise = output_dict['gt_noise']
        loss = F.mse_loss(pred_noise[part_valids], noise[part_valids])
        
        return {'mse_loss': loss}


    def training_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        
        mse_loss = loss_dict['mse_loss']
        self.log(f"training_loss/mse_loss", mse_loss, on_step=True, on_epoch=True)

        return mse_loss
    

    def validation_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        mse_loss = loss_dict['mse_loss']
        self.log(f"val/mse_loss", mse_loss, on_step=False, on_epoch=True)
        
        
        gt_trans = data_dict['part_trans']
        gt_rots = data_dict['part_rots']
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)
        noise_trans = randn_tensor(gt_trans_and_rots.shape, device=self.device)
   

        B, P, N, C = data_dict["part_pcs"].shape

        for t in tqdm(self.noise_scheduler.timesteps):
            timesteps = t.reshape(-1).repeat(len(noise_trans)).cuda()

            ############################################################################
            # apply rotation and translation to part pcs
            part_pcs = self._apply_rots(data_dict['part_pcs'], noise_trans)
            _, latent, xyz = self.encoder(part_pcs.flatten(0, 1).transpose(1, 2))
            xyz = xyz.reshape(B, P, self.num_points, 3)
            latent = latent.reshape(B, P, self.num_points, self.num_channels)
            # apply translation to fps xyz
            xyz = self._apply_trans(xyz, noise_trans)
            ############################################################################

            pred_noise = self.diffusion(noise_trans, timesteps, latent, xyz, data_dict["part_valids"])
            vNext = self.noise_scheduler.step(pred_noise, t, noise_trans).prev_sample
            noise_trans = vNext

        pts = data_dict['part_pcs']
        pred_translation = noise_trans[..., :3]

        pred_rots = noise_trans[..., 3:]

        acc = calc_part_acc(pts, trans1=pred_translation, trans2=gt_trans,
                            rot1=pred_rots, rot2=gt_rots, valids=data_dict['part_valids'], 
                            chamfer_distance=self.metric)
        
        rmse_r = rot_metrics(pred_rots, gt_rots, data_dict['part_valids'], 'rmse')
        rmse_t = trans_metrics(pred_translation, gt_trans,  data_dict['part_valids'], 'rmse')

        self.acc_list.append(torch.mean(acc))
        self.rmse_r_list.append(torch.mean(rmse_r))
        self.rmse_t_list.append(torch.mean(rmse_t))
        


    def on_validation_epoch_end(self):
        self.log(f"eval/part_acc", torch.mean(torch.stack(self.acc_list)))
        self.log(f"eval/rmse_r", torch.mean(torch.stack(self.rmse_r_list)))
        self.log(f"eval/rmse_t", torch.mean(torch.stack(self.rmse_t_list)))
        self.acc_list = []
        self.rmse_r_list = []
        self.rmse_t_list = []
        

    def test_step(self, data_dict, idx):
        self.validation_step(data_dict, idx)
        pass
    
    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
        return optimizer