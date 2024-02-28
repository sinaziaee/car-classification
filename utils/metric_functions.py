from sklearn.metrics import roc_auc_score
from sklearn import metrics
import torch
import pandas as pd
from tqdm import tqdm
from utils import util_functions, metric_functions
import json
import numpy as np


def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):
    
    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


def calculate_metrics(y_test, y_pred):
    macro_averaged_precision = metrics.precision_score(y_test, y_pred, average = 'macro')
    micro_averaged_precision = metrics.precision_score(y_test, y_pred, average = 'micro')

    macro_averaged_recall = metrics.recall_score(y_test, y_pred, average = 'macro')
    micro_averaged_recall = metrics.recall_score(y_test, y_pred, average = 'micro')

    macro_averaged_f1 = metrics.f1_score(y_test, y_pred, average = 'macro')
    micro_averaged_f1 = metrics.f1_score(y_test, y_pred, average = 'micro')
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    
    roc_auc_dict = roc_auc_score_multiclass(y_test, y_pred)
    
    return accuracy, macro_averaged_precision, micro_averaged_precision, macro_averaged_recall, micro_averaged_recall, macro_averaged_f1, micro_averaged_f1, roc_auc_dict

def train_fn(data_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    true_labels = []
    predicted_labels = []
    
    correct_classes = 0
    total_classes = 0
    for batch in data_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        # prediction
        outputs = model(images)
        # evaluating phase
        true_labels.append(labels.detach().cpu().numpy().tolist())
        predicted_labels.append(outputs.detach().cpu().numpy().tolist())
        optimizer.zero_grad() 
        loss = criterion(outputs, labels)
        with torch.no_grad():
            _, generalized_outputs = torch.max(outputs.data, 1)
            total_classes += labels.size(0)
            correct_classes += (generalized_outputs == labels).sum().item()   

        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct_classes / total_classes
    # acc, mac_precision, mic_precision, mac_recall, mic_recall, mac_f1, mic_f1, roc_auc_dict = metric_functions.calculate_metrics(true_labels, predicted_labels)
    return avg_loss, accuracy

def eval_fn(data_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    true_labels = []
    predicted_labels = []
    
    correct_classes = 0
    total_classes = 0
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
                
            outputs = model(images)
            _, generalized_outputs = torch.max(outputs.data, 1)
            # evaluating phase
            true_labels.append(labels.detach().cpu().numpy().tolist())
            predicted_labels.append(outputs.detach().cpu().numpy().tolist())
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            
            total_classes += labels.size(0)
            correct_classes += (generalized_outputs == labels).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100.0 * correct_classes / total_classes
        # acc, mac_precision, mic_precision, mac_recall, mic_recall, mac_f1, mic_f1, roc_auc_dict = metric_functions.calculate_metrics(true_labels, predicted_labels)
    return avg_loss, accuracy


def inference(model, data_loader, device=torch.device('cpu')):
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for images in data_loader:
            outputs = model(images)
            _, generalized_outputs = torch.max(outputs.data, 1)
            for generalized_output in generalized_outputs:
                predicted_labels.append(generalized_output.item())
    return predicted_labels

def perform_inference(model, save_results_path, test_labels_path, test_loader):
    results = inference(model, test_loader)
    test_labels_df = pd.read_csv(test_labels_path)
    test_labels_df['Predicted'] = results
    test_labels_df.to_csv(save_results_path, index=False)
    
    
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
        if scheduler is not None:
            scheduler.step()
        
        # Access the current learning rate
        if scheduler is not None:
            current_lr = scheduler.get_lr()[0]
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{results_folder}/best_model.pt')
            print('SAVED-MODEL')
        
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Valid Loss: {valid_loss}, Valid Acc: {valid_acc}')
        if epoch % 5 == 0:
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