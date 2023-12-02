from run_model import *
from get_data import *
import torch
import argparse
import pandas as pd
import json
from transforms import *
from torchvision.transforms import ToTensor, Resize, Compose, RandomAffine, ColorJitter, v2, ToPILImage, Lambda

# Reads command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('build_target', nargs='*', default='all')
parser.add_argument('--config', type=str, default='config.json')

args = vars(parser.parse_args())

def main(args):
    with open(args['config']) as f:
        config = json.load(f)
        
    best_model = None
    model, criterion, optimizer, scheduler, default_transforms = prepare_training(config)
    
    # Change train and test transforms
    if config['dataloaders']['use_custom_transforms']:
        train_transforms, test_transforms = get_transforms()
    else:
        train_transforms, test_transforms = default_transforms, default_transforms
    
    train_dl, val_dl, test_dl = get_dataloaders(config, 
                                                train_transform=train_transforms, 
                                                test_transform=test_transforms)

    dataloaders = {'train':train_dl, 'val':val_dl, 'test':test_dl}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    build_target = args['build_target']
    if 'all' in build_target:
        build_target = ['train_model', 'test_model']
        
    if 'train_model' in build_target:
        best_model, train_loss, val_loss = train_model(model, criterion, \
                                                       optimizer, scheduler, \
                                                       dataloaders, device, config)
        plot_loss(train_loss, val_loss, config)
        
    if 'test_model' in build_target:
        if best_model is None:
            model.load_state_dict(torch.load(config['filepaths']['saved_weights_path']))
            best_model = model
            
        test_loss, target_labs, pred_labs = test_model(best_model, criterion, device, dataloaders['test'])
        print(f'Test Loss: {test_loss}')

        plot_results(target_labs.squeeze(), pred_labs.squeeze(), config)
        plot_auc(target_labs.squeeze(), pred_labs.squeeze(), config)
        save_results(target_labs, pred_labs, config) 

if __name__ == "__main__":
    main(args)
