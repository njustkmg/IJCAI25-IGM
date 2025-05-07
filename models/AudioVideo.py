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


class AudioEncoder(nn.Module):
    def __init__(self, mask_model=1):
        super(AudioEncoder, self).__init__()
        self.mask_model = mask_model
        self.audio_net = resnet18(modality='audio')

    def forward(self, audio, step=0, balance=0, s=400, a_bias=0):
        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)  # [512,1]
        a = torch.flatten(a, 1)  # [512]
        return a


class VideoEncoder(nn.Module):
    def __init__(self, fps, mask_model=1):
        super(VideoEncoder, self).__init__()
        self.mask_model = mask_model
        self.video_net = resnet18(modality='visual')
        self.fps = fps

    def forward(self, video, step=0, balance=0, s=400, v_bias=0):
        v = self.video_net(video)
        (_, C, H, W) = v.size()
        B = int(v.size()[0] / self.fps)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        return v


class AVClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(AVClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(mask_model)
        self.video_encoder = VideoEncoder(config['fps'], mask_model)
        self.hidden_dim = 512
        self.cls_a = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )
        self.cls_v = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, config['setting']['num_class'])
        )
        self.fea_in = defaultdict(dict)
        self.P_matrix = defaultdict(dict)

    def forward(self, input_list):
        audio = input_list[0]
        video = input_list[1]
        state = input_list[2]
        if state == 'test_audio':
            a_feature = self.audio_encoder(audio)
            result_a = self.cls_a(a_feature)
            soft_a = F.softmax(result_a, dim=1)
            return soft_a
        elif state == 'test_video':
            v_feature = self.video_encoder(video)
            result_v = self.cls_v(v_feature)
            soft_v = F.softmax(result_v, dim=1)
            return soft_v
        elif state == 'test':
            a_feature = self.audio_encoder(audio)
            v_feature = self.video_encoder(video)
            result_a = self.cls_a(a_feature)
            result_v = self.cls_v(v_feature)
            soft_a = F.softmax(result_a, dim=1)
            soft_v = F.softmax(result_v, dim=1)
            return (soft_a + soft_v) / 2
        elif state == 'train_audio':
            a_feature = self.audio_encoder(audio)
            result_a = self.cls_a(a_feature)
            return a_feature, result_a
        elif state == 'train_video':
            v_feature = self.video_encoder(video)
            result_v = self.cls_v(v_feature)
            return v_feature, result_v

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
    model = AVClassifier(config=cfg)
    return model