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

class TextImage(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        #download the bert-base from huggingface
        # https://huggingface.co/google-bert/bert-base-uncased/tree/main
        #download the resnet from torchvision
        # https://download.pytorch.org/models/resnet50-0676ba61.pth
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        self.text_encoder = BertModel.from_pretrained('checkpoint/bert', add_pooling_layer=False)
        self.visual_encoder = torchvision.models.resnet50()
        checkpoint = torch.load('checkpoint/resnet50/resnet50-0676ba61.pth')
        self.visual_encoder.load_state_dict(checkpoint)
        self.cls_t=nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.Linear(64,config['setting']['num_class'])
        )
        self.cls_i=nn.Sequential(
            nn.Linear(self.visual_encoder.fc.out_features,self.text_encoder.config.hidden_size),
            nn.Linear(self.text_encoder.config.hidden_size,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.Linear(64,config['setting']['num_class'])
        )
        self.fea_in = defaultdict(dict)
        self.P_matrix = defaultdict(dict)
    def forward(self, input_list):
        text = input_list[0]
        image = input_list[1]
        state = input_list[2]
        if state == 'train_text':
            text_embeds = self.text_encoder(text.input_ids,
                                                 attention_mask=text.attention_mask,
                                                 return_dict=True
                                                 ).last_hidden_state[:,0,:]
            result = self.cls_t(text_embeds)
            return text_embeds,result
        if state == 'train_image':
            image_embeds = self.visual_encoder(image)
            result = self.cls_i(image_embeds)
            return image_embeds,result
        if state == 'test_text':
            text_embeds = self.text_encoder(text.input_ids,
                                                 attention_mask=text.attention_mask,
                                                 return_dict=True
                                                 ).last_hidden_state[:,0,:]
            result = self.cls_t(text_embeds)
            return result
        if state == 'test_image':
            image_embeds = self.visual_encoder(image)
            result = self.cls_i(image_embeds)
            return result
        if state == 'test':
            text_embeds = self.text_encoder(text.input_ids,
                                                 attention_mask=text.attention_mask,
                                                 return_dict=True
                                                 ).last_hidden_state[:,0,:]
            result_text= self.cls_t(text_embeds)
            image_embeds = self.visual_encoder(image)
            result_image= self.cls_i(image_embeds)
            soft_text=F.softmax(result_text,dim=1)
            soft_image=F.softmax(result_image,dim=1)
            return (soft_text+soft_image)/2

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
    def get_P_matrix(self, modules,fea_in,config):
        for m in modules:
            if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
                pass
            else:
                continue
            _, eigen_value, eigen_vector = torch.svd(fea_in[m.weight], some=False)
            max_eigen=torch.max(eigen_value)
            min_eigen=torch.min(eigen_value)
            learning_matrix=torch.exp(config['train']['temperature1']*(-(eigen_value-min_eigen)/(max_eigen-min_eigen)))

            learning_matrix=torch.diag(learning_matrix)
            self.P_matrix[m.weight]=eigen_vector@learning_matrix@eigen_vector.t()
    def update_cov(self, fea_in, k):
        cov = torch.mm(fea_in.transpose(0, 1), fea_in)
        if len(self.fea_in[k]) == 0:
            self.fea_in[k] = cov
        else:
            self.fea_in[k] = self.fea_in[k] + cov

def create_model(cfg):
    model = TextImage(config=cfg)

    return model