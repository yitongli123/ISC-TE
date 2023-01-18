import os
import json
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from scripts.dataset import get_transforms, dsbDataset
from segmentation_models_pytorch.decoders.unet import Unet
from scripts.utils import seed_everything
from Learner import Learner


parser = argparse.ArgumentParser() 
parser.add_argument('--config_path', type=str, help='config path') 
args = parser.parse_args() 
file_dir = args.config_path
with open(file_dir) as f:
        config = json.load(f) 
config = edict(config) 
config = config.TRAIN

if __name__ == '__main__':
    seed_everything(config.seed)

    model = Unet(encoder_name='resnet50', encoder_weights='imagenet', decoder_use_batchnorm=True,
                 decoder_attention_type='scse', classes=2, activation=None)
   
    df = pd.read_csv(config.df_path)
    train_df = df[df.fold != config.fold].reset_index(drop=True)
    valid_df = df[df.fold == config.fold].reset_index(drop=True)
    
    transforms = get_transforms(config.input_size, need=('train', 'val'))
    
    train_dataset = dsbDataset(config.data_dir, config.scr_dir, config.mask_dir, train_df,
                               tfms=transforms['train'], return_id=False)
    valid_dataset = dsbDataset(config.data_dir, config.scr_dir, config.mask_dir, valid_df,
                               tfms=transforms['val'], return_id=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, num_workers=config.num_workers,
                              shuffle=False)
    
  
    Learner = Learner(model, train_loader, valid_loader, config)
        
    Learner.fit(config.n_epochs)
