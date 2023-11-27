import torch.nn as nn
import torch
from torch.nn import functional as F
from jigsaw_latent.model.model_utils import (
    timestep_embedding,
    PositionalEncoding,
    EmbedderNerf
)
from jigsaw_latent.model.transformer import EncoderLayer
from pytorch3d import transforms


class DiffModel(nn.Module):
    """
  Transformer-based diffusion model
  """

    def __init__(self, cfg):
        super(DiffModel, self).__init__()


        self.model_channels = cfg.model.embed_dim
        self.out_channels = cfg.model.out_channels
        self.position_encoding = cfg.model.position_encoding

        self.num_layers = cfg.model.num_layers
        self.num_heads = cfg.model.num_heads

        self.activation = nn.SiLU()
        self.transformer_layers = nn.ModuleList(
            [EncoderLayer(self.model_channels, self.num_heads, 0.1, self.activation) for x in range(self.num_layers)])

        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.model_channels),
            nn.SiLU(),
            nn.Linear(self.model_channels, self.model_channels),
        )

        # init learnable embedding
        self.num_queries = cfg.data.max_num_part
        self.learnable_embedding = nn.Embedding(self.num_queries, self.model_channels)

        multires = 10
        embed_kwargs = {
            'include_input': True,
            'input_dims': 7,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        
        embedder_obj = EmbedderNerf(**embed_kwargs)
        self.param_embedding = lambda x, eo=embedder_obj: eo.embed(x)
        self.param_fc = nn.Linear(147, self.model_channels)


        multires = 10
        embed_pos_kwargs = {
            'include_input': True,
            'input_dims': 3,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        embedder_pos = EmbedderNerf(**embed_pos_kwargs)
        # Pos embedding for positions of points xyz
        self.pos_embedding = lambda x, eo=embedder_pos: eo.embed(x)
        self.pos_fc = nn.Linear(63, self.model_channels)

        self.shape_embedding = nn.Linear(4, self.model_channels)

        # Pos encoding for indicating the sequence. which part is for the reference
        self.pos_encoding = PositionalEncoding(self.model_channels)

        self.out = nn.Linear(self.model_channels, self.out_channels)

        self.output_linear1 = nn.Linear(self.model_channels, self.model_channels)
        self.output_linear2 = nn.Linear(self.model_channels, self.model_channels // 2)
        self.output_linear3 = nn.Linear(self.model_channels // 2, self.out_channels)


    def _gen_mask(self, L, N, B, mask):
        self_block = torch.ones(L, L, device=mask.device)  # Each L points should talk to each other
        self_mask = torch.block_diag(*([self_block] * N))  # Create block diagonal tensor
        self_mask = self_mask.unsqueeze(0).repeat(B, 1, 1)  # Expand dimensions to [B, N*L, N*L]

        flattened_mask = mask.unsqueeze(-1).repeat(1, 1, L).flatten(1, 2)  # shape [B, N*L]
        flattened_mask = flattened_mask.unsqueeze(1)  # shape [B, 1, N*L]
        gen_mask = flattened_mask * flattened_mask.transpose(-1, -2)  # shape [B, N*L, N*L]
        return self_mask, gen_mask
    

    def _gen_cond(self, timesteps, x, xyz, latent):
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        time_emb = time_emb.unsqueeze(1)

        x = x.flatten(0, 1)  # (B*N, 7)

        xyz = xyz.flatten(0, 1)  # (B*N, L, 3)

        latent = latent.flatten(0, 1)  # (B*N, L, 4)

        x_emb = self.param_fc(self.param_embedding(x))

        shape_emb = self.shape_embedding(latent)
        xyz_pos_emb = self.pos_fc(self.pos_embedding(xyz))

        return x_emb, shape_emb, xyz_pos_emb, time_emb
    

    def _out(self, data_emb, B, N, L):
        out = data_emb.reshape(B, N, L, self.model_channels)
        # Avg pooling
        out = out.mean(dim=2)
        out_dec = self.output_linear1(out)
        out_dec = self.activation(out_dec)
        out_dec = self.output_linear2(out_dec)
        out_dec = self.output_linear3(out_dec)
        return out_dec

    def forward(self, x, timesteps, latent, xyz, part_valids):
        """
        Latent already transform

        forward pass 
        x : (B, N, 7)
        timesteps : (B, 1)
        latent : (B, N, L, 4)
        xyz : (B, N, L, 3)
        mask: B, N
        """

        B, N, L, _ = latent.shape

        x_emb, shape_emb, pos_emb, time_emb = self._gen_cond(timesteps, x, xyz, latent)

        self_mask, gen_mask = self._gen_mask(L, N, B, part_valids)

        # # pe
        pe = torch.zeros_like(x_emb.reshape(B, N, -1)) 
        pe = self.pos_encoding(pe)  # B, N, C
        pe = pe.reshape(B, N, 1, -1).repeat(1, 1, L, 1)  # B, N, L, C


        x_emb = x_emb.reshape(B, N, 1, -1)
        x_emb = x_emb.repeat(1, 1, L, 1)
        
        condition_emb = shape_emb.reshape(B, N*L, -1) + \
                            pos_emb.reshape(B, N*L, -1) + time_emb \
                            + pe.reshape(B, N*L, -1)
        
        # B, N*L, C
        data_emb = x_emb.reshape(B, N*L, -1) 

        data_emb = data_emb + condition_emb


        for layer in self.transformer_layers:
            data_emb = layer(data_emb, self_mask, gen_mask)
            

        # data_emb (B, N*L, C)
        out_dec = self._out(data_emb, B, N, L)

        return out_dec
