from sklearn.metrics import roc_auc_score
from sklearn import metrics
import torch

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