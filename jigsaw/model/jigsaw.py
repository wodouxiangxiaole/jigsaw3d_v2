import torch
from torch.nn import functional as F
import lightning.pytorch as pl
import hydra
from jigsaw.model.diffusion import DiffModel
from diffusers import DDPMScheduler
from tqdm import tqdm
from chamferdist import ChamferDistance
from jigsaw.evaluation.jigsaw_evaluator import (
    calc_part_acc,
    rot_metrics,
    trans_metrics,
    randn_tensor
)


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

        self.acc_list = []
        self.rmse_r_list = []
        self.rmse_t_list = []

        self.metric = ChamferDistance()


    def forward(self, data_dict):
        gt_trans = data_dict['part_trans']
        gt_rots = data_dict['part_quat']
        gt_rots_trans = torch.cat([gt_trans, gt_rots], dim=-1)

        noise = torch.randn(gt_rots_trans.shape, device=self.device)

        B = gt_trans.shape[0]

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,),
                                  device=self.device).long()
        
        noisy_trans = self.noise_scheduler.add_noise(gt_rots_trans, noise, timesteps)

        pred_noise = self.diffusion(noisy_trans, timesteps, data_dict)

        output_dict = {
            'pred_noise': pred_noise,
            'noise': noise
        }

        return output_dict


    def _loss(self, data_dict, output_dict):
        pred_noise = output_dict['pred_noise']
        part_valids = data_dict['part_valids'].bool()
        noise = output_dict['noise']
        loss = F.mse_loss(pred_noise, noise[part_valids])
        
        return {'mse_loss': loss}


    def training_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        
        mse_loss = loss_dict['mse_loss']
        self.log(f"training_loss/mse_loss", mse_loss, on_step=False, on_epoch=True)
        total_loss = mse_loss

        return total_loss
    

    def validation_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        mse_loss = loss_dict['mse_loss']
        self.log(f"val/mse_loss", mse_loss, on_step=False, on_epoch=True)
        # gt_trans = data_dict['part_trans']
        # gt_rots = data_dict['part_quat']
        # trans = torch.cat([gt_trans, gt_rots], dim=-1)
        # noise_trans = randn_tensor(trans.shape, device=self.device)
        # valids = data_dict['part_valids']

        # for t in tqdm(self.noise_scheduler.timesteps):
        #     timesteps = t.reshape(-1).repeat(len(noise_trans)).cuda()

        #     pred_noise = self.diffusion(noise_trans, timesteps, data_dict)
        #     valid_indices = valids.flatten().nonzero(as_tuple=False).squeeze()
        #     result = torch.zeros(noise_trans.shape[0]*noise_trans.shape[1], 7).cuda()
        #     result[valid_indices] = pred_noise
        #     vNext = self.noise_scheduler.step(result.reshape(noise_trans.shape[0], noise_trans.shape[1], -1), 
        #                                                      t, noise_trans).prev_sample
        #     noise_trans = vNext

        # pts = data_dict['part_pcs']
        # pred_translation = noise_trans[..., :3]
        # gt_translation = gt_trans[..., :3]

        # pred_rots = noise_trans[..., 3:]
        # gt_rots = gt_trans[..., 3:]

        # acc = calc_part_acc(pts, trans1=pred_translation, trans2=gt_translation,
        #                     rot1=pred_rots, rot2=gt_rots, valids=valids, 
        #                     chamfer_distance=self.metric)
        
        # rmse_r = rot_metrics(pred_rots, gt_rots, valids, 'rmse')
        # rmse_t = trans_metrics(pred_translation, gt_translation, valids, 'rmse')

        # self.acc_list.append(torch.mean(acc))
        # self.rmse_r_list.append(torch.mean(rmse_r))
        # self.rmse_t_list.append(torch.mean(rmse_t))
        


    def on_validation_epoch_end(self):
        # self.log(f"eval/part_acc", torch.mean(torch.stack(self.acc_list)))
        # self.log(f"eval/rmse_r", torch.mean(torch.stack(self.rmse_r_list)))
        # self.log(f"eval/rmse_t", torch.mean(torch.stack(self.rmse_t_list)))
        # self.acc_list = []
        # self.rmse_r_list = []
        # self.rmse_t_list = []
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

    def on_test_epoch_end(self):
        pass
