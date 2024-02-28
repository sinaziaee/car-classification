import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from torch import optim
import json
import torch.optim.lr_scheduler as lr_scheduler
import os

from tqdm import tqdm
from utils import util_functions, metric_functions, custom_dataset

train_labels_path = os.path.join('dataset', 'train.csv')
test_labels_path = os.path.join('dataset', 'test.csv')

train_images_dir = os.path.join('dataset', 'train')
test_images_dir = os.path.join('dataset', 'test')

train_labels_df = pd.read_csv(train_labels_path)
test_labels_df = pd.read_csv(test_labels_path)

train_transformer = transforms.Compose([transforms.Resize((400, 400)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transformer = transforms.Compose([transforms.Resize((400, 400)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

## Checking how many different classes we have
no_classes = len(list(train_labels_df['label'].unique()))

dataset = custom_dataset.CustomDataset(images_dir=train_images_dir, 
                                                df=train_labels_df, transforms=test_transformer)
augmented_dataset = custom_dataset.CustomDataset(images_dir=train_images_dir, 
                                                df=train_labels_df, transforms=train_transformer)
# Define the sizes for the training and validation sets
train_size = int(0.8 * len(dataset))  # 80% of the data for training
val_size = len(dataset) - train_size   # Remaining 20% for validation

# Use random_split to create training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset = ConcatDataset([train_dataset, augmented_dataset])
print(len(train_dataset), len(val_dataset))
BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train_loop(n_epochs, model, optimizer, train_loader, valid_loader, device,
                                criterion, scheduler=None):

    model = model.to(device)
        
    best_valid_loss = np.Inf

    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    result_directory = 'results'
    results_folder = util_functions.create_result_folder_by_date_and_time_folder(result_directory)
    for epoch in tqdm(range(n_epochs)):
        train_loss, train_acc = metric_functions.train_fn(data_loader=train_loader, model=model, criterion=criterion, 
                                optimizer=optimizer, device=device)
        valid_loss, valid_acc = metric_functions.eval_fn(data_loader=valid_loader, model=model, criterion=criterion,
                                        device=device)
        
        scheduler.step()
        
        # Access the current learning rate
        current_lr = scheduler.get_lr()[0]
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{results_folder}/best_model.pt')
            print('SAVED-MODEL')
        
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Valid Loss: {valid_loss}, Valid Acc: {valid_acc}, lr: {current_lr}')
        if epoch % 2 == 0:
            util_functions.visualize_training(train_loss_list=train_loss_list, valid_loss_list=valid_loss_list,
                                            train_acc_list=train_acc_list, valid_acc_list=valid_acc_list, results_folder=results_folder)
            
        lists_dict = {
            'train_loss_list': train_loss_list,
            'train_acc_list': train_acc_list,
            'valid_loss_list': valid_loss_list,
            'valid_acc_list': valid_acc_list,
        }

        with open(f'{results_folder}/training_trend.json', 'w') as f:
            json.dump(lists_dict, f)
        
    return f'{results_folder}/best_model.pt'

torch.cuda.empty_cache()
device = torch.device('cuda:0')
# load pretrained dataset
# model = models.resnet50(pretrained=True)
model = models.resnet34(pretrained=True)
# changing the last layer to our cause
model.fc = nn.Linear(model.fc.in_features, no_classes)
model.to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
# scheduler = lr_scheduler.ReduceLROnPlateau()
n_epochs = 51

result_folder = train_loop(n_epochs, model, optimizer, train_loader, val_loader, device, criterion, scheduler=scheduler)