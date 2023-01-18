import os
import json
import numpy as np
import pandas as pd
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from easydict import EasyDict as edict
from scripts.dataset import get_transforms, dsbTestDataset
from segmentation_models_pytorch.decoders.unet import Unet
from scripts.metric_mdice import Evaluator as mdice_evaluator
from scripts.metric import Evaluator as iou_evaluator


parser = argparse.ArgumentParser() 
parser.add_argument('--config_path', type=str, help='config path') 
args = parser.parse_args() 
file_dir = args.config_path
with open(file_dir) as f:
        config = json.load(f) 
config = edict(config) 
config = config.TEST


def inference_image(net, images):
    with torch.no_grad():
        predictions = net(images)
        predictions = F.softmax(predictions, dim=1)
    return predictions.detach().cpu().numpy()


def inference(net, test_loader, save_dir=None):
    semantic_eval, instance_eval = iou_evaluator(), mdice_evaluator()
    semantic_eval.reset()
    instance_eval.reset()
    for image_names, images, masks in tqdm(test_loader):
        images = images.to(torch.device(config.device))
        masks = masks.numpy()
        saliency = inference_image(net, images)
        predictions = np.argmax(saliency, axis=1).astype('uint8')
        semantic_eval.add_batch((masks > 0).astype('uint8'), predictions)
        for image_name, pred, mask in zip(image_names, saliency, masks):
            instance_eval.add_pred(mask, pred[1])
            if save_dir:
                Image.fromarray((pred[1] * 255).astype('uint8')).save(os.path.join(save_dir, f'{image_name}.png'))
    return semantic_eval.IoU, instance_eval.Dice


if __name__ == '__main__':
    model = Unet(encoder_name='resnet50', encoder_weights='imagenet', decoder_use_batchnorm=True,
                 decoder_attention_type='scse', classes=2, activation=None)

    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(torch.device(config.device))
    model.eval()
    print(f'Model Loaded: {config.model_path}')

    test_df = pd.read_csv(config.df_path)
    transforms = get_transforms(config.input_size, need=('val'))

    test_dataset = dsbTestDataset(config.data_dir, config.mask_dir, test_df,
                                  tfms=transforms['val'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                             shuffle=False, sampler=None, pin_memory=True)

    if config.save_result:
        save_dir = os.path.join(os.path.dirname(config.model_path), 'predictions')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    iou, mdice = inference(model, test_loader, save_dir)
    print(f'IoU: {iou:.4f}, mDice: {mdice:.4f}')
