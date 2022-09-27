import torch.nn as nn
import torch
from encoder import Projector
from backbone import VitSegNet
from head import GridSeg
from config import Config


class Detector(nn.Module):
    def __init__(self,
                cfg=None):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(144, 144, 3, 3)
        self.pcencoder = Projector(resnet='resnet34')
        self.backbone = VitSegNet(depth=3)
        self.heads = GridSeg()

    def forward(self, points):
        '''
        input:
            # points: torch.Size([1, 3, 1152, 1152])
        '''
        output = {}
        fea = self.pcencoder(points)   
        
        mlp, fea_backbone = self.backbone(fea)
        # fea: torch.Size([1, 8, 144, 144])

        return mlp, fea_backbone

        out = self.heads(fea_backbone)  

        return out, fea_backbone, fea

    def get_loss(self, out, label):
        output = {}
        output.update(self.heads.loss(out, label))

        return output

    def get_lanes(self, out, label):
        result = {}
        result.update({'conf': out[:,7,:,:], 'cls': out[:,:7,:,:]})
        result.update({
            'lane_maps': self.heads.get_lane_map_numpy_with_label(
                                result, label, is_img=self.cfg.view)})

        return result

if __name__=="__main__":
    path_config = './network_config.py'
    path_split = path_config.split('/')
    cfg = Config.fromfile(path_config)
    net = Detector(cfg)
    print(net)
