import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from datasets.sarcasm_dataset import sarcasm_dataset
from datasets.twitter_dataset import twitter_dataset
from datasets.randaugment import RandomAugment

def create_dataset(dataset, config):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    if dataset =='sarcasm':
        train_dataset = sarcasm_dataset(config['train_file'], train_transform, config['image_root'],config['max_tokens'])
        test_dataset = sarcasm_dataset(config['test_file'], test_transform, config['image_root'],config['max_tokens'])
        return train_dataset, test_dataset
    elif dataset == 'twitter':
        train_dataset = twitter_dataset(config['train_file'], train_transform, config['image_root'],config['max_tokens'])
        test_dataset = twitter_dataset(config['test_file'], test_transform, config['image_root'],config['max_tokens'])
        return train_dataset, test_dataset