import torch.nn as nn
import torch.nn.functional as F
from jigsaw_latent.model.modules.pn2_utils import PointNetSetAbstraction
import pdb

    

class PN2(nn.Module):
    def __init__(self):
        super(PN2, self).__init__()
        in_channel = 3
        
        self.num_point = 27
        self.num_dim = 4
        self.sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=self.num_point, radius=0.8, nsample=64, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)

        self.conv6 = nn.Conv1d(in_channels=512, out_channels=self.num_dim, kernel_size=1)

        self.fc1 = nn.Linear(self.num_point*self.num_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1000*3)


    def forward(self, xyz):
        # B, C, L
        B, _, _ = xyz.shape
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)  
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)    # [B, C, L]

        global_feat = self.conv6(l3_points) # points x dim
        
        x = global_feat.view(B, self.num_point * self.num_dim)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x).reshape(B, 1000, 3)

        return x, global_feat.permute(0,2,1), l3_xyz.permute(0, 2, 1)   # [N, L, C]   

    def decode(self, global_feat):
        """
        Input: [N, L, C]
        """
        global_feat = global_feat.permute(0,2,1)  # [N, C, L]
        B, _, _ = global_feat.shape
        x = global_feat.reshape(B, self.num_point*self.num_dim) 
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x).reshape(B, 1000, 3)
        
        return x
    
