import json
import os
import random
import re
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption
def clean_str(text):
    english_words = re.findall(r'\b[A-Za-z]+\b', text)
    result_text = ' '.join(english_words)
    return result_text

class twitter_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=50):
        self.ann = pd.read_csv(ann_file,sep='\t')
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}
        self.cls_num = 3

    def __len__(self):
        return len(self.ann)

    def get_num_classes(self):
        return self.cls_num

    def __getitem__(self, index):
        label = self.ann["Label"][index]
        label = int(label)

        text = self.ann['String'][index]
        text = clean_str(text)
        text = pre_caption(text,self.max_words)

        imagePath = self.image_root + self.ann["ImageID"][index]
        image = Image.open(imagePath).convert('RGB')
        image = self.transform(image)

        return image, text, label