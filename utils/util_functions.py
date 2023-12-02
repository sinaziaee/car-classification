from datetime import datetime
import os
import cv2 as cv
import matplotlib.pyplot as plt
import torch


def create_folder(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
    return path

def create_result_folder_by_date_and_time_folder(base_path):
    create_folder(base_path)
    current_datetime = datetime.now()
    folder_name = current_datetime.strftime("%Y-%m-%d_%H-%M")
    new_path = os.path.join(base_path, folder_name)
    create_folder(new_path)
    return new_path

def visualize_image(img_path, lbl):
    samp_img_rgb = cv.cvtColor(cv.imread(img_path, 3), cv.COLOR_BGR2RGB)
    plt.title(f'Car class {lbl}')
    plt.imshow(samp_img_rgb)
    plt.show()
    
def check_no_of_each_class(train_labels_df):
    class_counts = train_labels_df['label'].value_counts()
    class_counts.sort_values(ascending=True, inplace=True)
    class_counts.sort_index(ascending=True, inplace=True)
    plt.figure(figsize=(20, 5))
    plt.bar(class_counts.index, class_counts.values)
    plt.xlabel('Car Class')
    plt.ylabel('No. of each machine')
    plt.show()
    print(f'Min: {class_counts.min()}, Max: {class_counts.max()}')
    print(class_counts)
    
def visualize_training(train_loss_list, valid_loss_list, train_acc_list=None, valid_acc_list=None, results_folder= None):
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    ax[0].plot(train_loss_list)
    ax[0].plot(valid_loss_list)
    ax[0].set_title("Train-Valid Loss")
    ax[0].legend()
    ax[1].plot(train_acc_list)
    ax[1].set_title("Train Accuracy")
    ax[1].plot(valid_acc_list)
    ax[1].legend()
    plt.savefig(f'{results_folder}/train_result_fig.png')
    plt.legend()
    plt.show()
    
def calculate_mean_std_of_dataset(dataset, is_test=False, with_path=False):
    mean_sum = torch.zeros(3)
    std_sum = torch.zeros(3)
    if is_test == False:
        if with_path == False:
            for image, _ in dataset:
                mean = torch.mean(image, dim = (1, 2))
                std = torch.std(image, dim = (1, 2))
                mean_sum += mean
                std_sum += std
        else:
            for image, _, _ in dataset:
                mean = torch.mean(image, dim = (1, 2))
                std = torch.std(image, dim = (1, 2))
                mean_sum += mean
                std_sum += std            
    else:
        for image in dataset:
            mean = torch.mean(image, dim = (1, 2))
            std = torch.std(image, dim = (1, 2))
            mean_sum += mean
            std_sum += std
    num_samples = len(dataset)
    mean = mean_sum / num_samples
    std = std_sum / num_samples
    return mean, std
