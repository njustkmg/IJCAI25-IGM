from utils import *
from os import path
from collections import OrderedDict
import torchvision
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.DotProductClassifier import create_cls
class ConcatFusion(nn.Module):
    def __init__(self,input_dim_text=768,input_dim_image=512,output_dim=2):
        super(ConcatFusion,self).__init__()
        self.text_fc=nn.Linear(input_dim_text,output_dim,bias=False)
        self.image_fc=nn.Linear(input_dim_image,output_dim,bias=False)
        self.bias=nn.Parameter(torch.rand(1,2),requires_grad=True)
    def forward(self,x,y):
        output_x=self.text_fc(x)
        output_y=self.image_fc(y)
        output=output_x+output_y+self.bias
        return x,y,output
class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.text_encoder = BertModel.from_pretrained('checkpoint/berthug',add_pooling_layer=False)
        self.visual_encoder = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.visual_encoder.fc.in_features
        self.visual_encoder.fc =  nn.Linear(num_features, 512)
        self.fusion_module=ConcatFusion(input_dim_image=self.visual_encoder.fc.out_features,input_dim_text=self.text_encoder.config.hidden_size)
    def forward(self, input_list):
        text=input_list[0]
        image=input_list[1]
        state=input_list[2]
        text_embeds = self.text_encoder(text.input_ids,
                            attention_mask = text.attention_mask,
                            return_dict = True
                            )        
        image_embeds=self.visual_encoder(image)
        
        text_embeds,image_embeds,output=self.fusion_module(text_embeds.last_hidden_state[:,0,:],image_embeds)
        if state=='train':
            return text_embeds,image_embeds,output
        else:
            return output
def create_model(cfg, use_fc=False, pretrain=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False ,*args):
    
    model = ALBEF(config=cfg, text_encoder=cfg['text_encoder'])

    
    return model