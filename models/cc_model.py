import os

import torch

import torch.nn as nn
class HRNetCrowdCounting(nn.Module):

    def __init__(self, model_dir: str, **kwargs):
        #super().__init__(model_dir, **kwargs)
        super(HRNetCrowdCounting, self).__init__()

        from .hrnet_aspp_relu import HighResolutionNet as HRNet_aspp_relu

        domain_center_model = os.path.join(
            model_dir, 'average_clip_domain_center_54.97.npz')
        net = HRNet_aspp_relu(
            attn_weight=1.0,
            fix_domain=0,
            domain_center_model=domain_center_model)
        net.load_state_dict(
            torch.load(
                os.path.join(model_dir, 'DCANet_final.pth'),
                map_location='cpu'))
        self.model = net

    def forward(self, inputs):
        return self.model(inputs)

