from utils import *
from os import path
from collections import OrderedDict
import torchvision
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict
from models.Resnet import resnet18

import torch.nn as nn
import torch
from models.Model3D import InceptionI3d

class RGBEncoder(nn.Module):
    def __init__(self, config):
        super(RGBEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=3)
        #download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('checkpoint/rgb_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.rgbmodel = model

    def forward(self, x):
        out = self.rgbmodel(x)
        return out  # BxNx2048


class OFEncoder(nn.Module):
    def __init__(self, config):
        super(OFEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=2)
        #download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('checkpoint/flow_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.ofmodel = model

    def forward(self, x):
        out = self.ofmodel(x)
        return out  # BxNx2048


class DepthEncoder(nn.Module):
    def __init__(self, config):
        super(DepthEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=1)
        #download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('checkpoint/rgb_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.depthmodel = model

    def forward(self, x):
        out = self.depthmodel(x)
        return out  # BxNx2048
class JointClsModel(nn.Module):
    def __init__(self, config):
        super(JointClsModel, self).__init__()
        self.rgb_encoder = RGBEncoder(config)
        self.of_encoder = OFEncoder(config)
        self.depth_encoder = DepthEncoder(config)
        self.hidden_dim = 1024
        self.cls_r = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )
        self.cls_o = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )
        self.cls_d = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )
        self.fea_in = defaultdict(dict)
        self.P_matrix = defaultdict(dict)
    def forward(self, input_list):
        rgb = input_list[0]
        of = input_list[1]
        depth = input_list[2]
        state = input_list[3]
        if state == 'train_rgb':
            rgb_feat = self.rgb_encoder(rgb)
            result_r = self.cls_r(rgb_feat)
            return rgb_feat,result_r
        elif state == 'train_of':
            of_feat = self.of_encoder(of)
            result_o = self.cls_o(of_feat)
            return of_feat,result_o
        elif state == 'train_depth':
            depth_feat = self.depth_encoder(depth)
            result_d = self.cls_d(depth_feat)
            return depth_feat,result_d
        elif state == 'test_rgb':
            rgb_feat = self.rgb_encoder(rgb)
            result_r = self.cls_r(rgb_feat)
            soft_r = F.softmax(result_r, dim=1)
            return soft_r
        elif state == 'test_of':
            of_feat = self.of_encoder(of)
            result_o = self.cls_o(of_feat)
            soft_o =F.softmax(result_o,dim=1)
            return soft_o
        elif state == 'test_depth':
            depth_feat = self.depth_encoder(depth)
            result_d = self.cls_d(depth_feat)
            soft_d = F.softmax(result_d,dim=1)
            return soft_d
        elif state == 'test':
            rgb_feat = self.rgb_encoder(rgb)
            result_r = self.cls_r(rgb_feat)
            soft_r = F.softmax(result_r, dim=1)

            of_feat = self.of_encoder(of)
            result_o = self.cls_o(of_feat)
            soft_o =F.softmax(result_o,dim=1)

            depth_feat = self.depth_encoder(depth)
            result_d = self.cls_d(depth_feat)
            soft_d = F.softmax(result_d,dim=1)
            return (soft_d+soft_o+soft_r)/3


    def compute_cov(self, module, fea_in, fea_out):
        if isinstance(module, nn.Linear):
            self.update_cov(torch.mean(fea_in[0], 0, True), module.weight)

        elif isinstance(module, nn.Conv2d):
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding

            fea_in_ = F.unfold(
                torch.mean(fea_in[0], 0, True), kernel_size=kernel_size, padding=padding, stride=stride)

            fea_in_ = fea_in_.permute(0, 2, 1)
            fea_in_ = fea_in_.reshape(-1, fea_in_.shape[-1])
            self.update_cov(fea_in_, module.weight)

        torch.cuda.empty_cache()
        return None
    #
    def get_P_matrix(self, modules, fea_in, config):
        for m in modules:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                pass
            else:
                continue
            _, eigen_value, eigen_vector = torch.svd(fea_in[m.weight], some=False)
            max_eigen = torch.max(eigen_value)
            min_eigen = torch.min(eigen_value)
            learning_matrix = torch.exp(
                config['train']['temperature1'] * (-(eigen_value - min_eigen) / (max_eigen - min_eigen)))

            learning_matrix = torch.diag(learning_matrix)
            self.P_matrix[m.weight] = eigen_vector @ learning_matrix @ eigen_vector.t()

    def update_cov(self, fea_in, k):
        cov = torch.mm(fea_in.transpose(0, 1), fea_in)
        if len(self.fea_in[k]) == 0:
            self.fea_in[k] = cov
        else:
            self.fea_in[k] = self.fea_in[k] + cov


def create_model(cfg):
    model = JointClsModel(config=cfg)
    return model