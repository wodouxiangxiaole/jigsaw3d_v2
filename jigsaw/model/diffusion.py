import torch.nn as nn
import torch
from torch.nn import functional as F
from jigsaw.model.model_utils import (
    timestep_embedding,
    get_batch_length_from_part_points,
    PositionalEncoding,
    EmbedderNerf
)
from jigsaw.model.transformer import EncoderLayer
from pytorch3d import transforms
from jigsaw.model.modules.pointnet2_msg import PointNet2PTMSGDynamic
from jigsaw.model.modules.point_transformer import (
    PointTransformerLayer,
    CrossAttentionLayer,
    TransitionDown
)
from pytorch3d import transforms


"""
Transformer-based diffusion model
"""
class DiffModel(nn.Module):

    def __init__(self, cfg):
        super(DiffModel, self).__init__()

        self.pc_feat_dim = cfg.model.PC_FEAT_DIM

        self.out_channel = cfg.model.out_channels

        self.encoder = PointNet2PTMSGDynamic(feat_out=128)

        self.model_channels = cfg.model.embed_dim
        self.tf_self1 = PointTransformerLayer(
            in_feat=self.pc_feat_dim, out_feat=self.pc_feat_dim,
            n_heads=8, nsampmle=16,
        )
        
        self.tf_cross1 = CrossAttentionLayer(d_in=self.pc_feat_dim,
                                             n_head=8,)
        

        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.model_channels),
            nn.SiLU(),
            nn.Linear(self.model_channels, self.model_channels),
        )

        self.transition_down = TransitionDown(in_planes=self.pc_feat_dim,
                                              out_planes=self.pc_feat_dim,
                                              stride=4)
        
        self.tf_layers = [("self", self.tf_self1), ("cross", self.tf_cross1), ("down", self.transition_down)]

        self.out = nn.Sequential(
            nn.Linear(self.model_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, self.out_channel)
        )

    
    
    def _extract_part_feats(self, part_pcs, batch_length):
        B, N_sum, _ = part_pcs.shape  # [B, N_sum, 3]
        # shared-weight encoder
        valid_pcs = part_pcs.reshape(B * N_sum, -1)
        valid_feats = self.encoder(valid_pcs, batch_length)  # [B * N_sum, F]
        pc_feats = valid_feats.reshape(B, N_sum, -1)  # [B, N_sum, F]
        return pc_feats

    
    def forward(self, noise_param, timesteps, data_dict):
        """
        noise_trans: [B, N_sum, 4]
        """
        part_valids = data_dict["part_valids"]

        noise_param = noise_param.reshape(-1, 7)

        noise_param = noise_param[part_valids.reshape(-1).bool()]
        noise_quat = noise_param[:, 3:]
        noise_quat = noise_quat / noise_quat.norm(dim=-1, keepdim=True)

        noise_trans = noise_param[:, :3]

        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        n_valid = torch.sum(part_valids, dim=1).to(torch.long)  # [B]
        n_pcs = data_dict["n_pcs"]  # [B, P]

        batch_length = get_batch_length_from_part_points(n_pcs, n_valids=n_valid).to(
            n_pcs.device
        )

        part_pcs = data_dict["part_pcs"]  # [B, N_sum, 3]
        
        reshape_part_pcs = part_pcs.reshape(-1, 3).contiguous()

        trans_part_pcs = []


        cnt = 0
        for i in range(batch_length.shape[0]):

            t = transforms.quaternion_apply(
                noise_quat[i], 
                reshape_part_pcs[cnt:cnt+batch_length[i], :]
                )
            t += t + noise_trans[i]
            trans_part_pcs.append(t)
            cnt += batch_length[i]
            

        part_pcs = torch.cat(trans_part_pcs, dim=0).reshape(-1, 5000, 3)
        part_input = part_pcs

        
        B, N_sum, _ = part_pcs.shape
        

        part_feats = self._extract_part_feats(part_input, batch_length)

        part_feats = part_feats + time_emb.unsqueeze(1)

        part_pcs_flatten = part_pcs.reshape(-1, 3).contiguous()

        for name, layer in self.tf_layers:
            if name == "self":
                part_feats = (
                    layer(
                        part_pcs_flatten,
                        part_feats.view(-1, self.pc_feat_dim),
                        batch_length,
                    )
                    .view(B, N_sum, -1)
                    .contiguous()
                )
            elif name == "cross":
                part_feats = layer(part_feats)
            else:
                part_pcs_flatten, part_feats, batch_length = self.transition_down(
                    part_pcs_flatten,
                    part_feats.view(-1, self.pc_feat_dim), 
                    batch_length
                )
        
        x = []

        cnt = 0
        for i in range(batch_length.shape[0]):

            x_b = part_feats[cnt:cnt+batch_length[i], :]
            x_b = torch.mean(x_b, dim=0, keepdim=True)
            x.append(x_b)
            cnt += batch_length[i]

            
        
        x = torch.cat(x, dim=0)
        out = self.out(x)


        return out
