import os
import sys
import json
import pickle
import random

import torch
from torch import nn
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.nn.functional as F


def CrossEntropy(outputs, targets):
    F.kl_div
    log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
    softmax_targets = F.softmax(targets/3.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # Ensure reproducibility of random results
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # Traverse folders, where each folder corresponds to a category
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # Sort to ensure consistent order
    flower_class.sort()
    # Generate class names and corresponding numeric indices
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # Store all training set image paths
    train_images_label = []  # Store corresponding indices for training images
    # val_images_path = []  # Store all validation set image paths
    # val_images_label = []  # Store corresponding indices for validation images
    every_class_num = []  # Store the total number of samples per category
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # Supported file extensions
    # Traverse files in each folder
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # Collect paths of all supported file types
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # Get index corresponding to this category
        image_class = class_indices[cla]
        # Record the number of samples in this category
        every_class_num.append(len(images))
        # Randomly sample validation data based on the given ratio
        # val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            # if img_path in val_path:  # If the path is in the sampled validation set, store it
            #     val_images_path.append(img_path)
            #     val_images_label.append(image_class)
            # else:  # Otherwise, store it in the training set
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    # print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # Plot a bar chart showing the number of images per category
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # Replace x-axis labels (0,1,2,3,4) with category names
        plt.xticks(range(len(flower_class)), flower_class)
        # Add value labels to the bars
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # Set x-axis label
        plt.xlabel('image class')
        # Set y-axis label
        plt.ylabel('number of images')
        # Set title of the bar chart
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # Reverse Normalize operation
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # Remove x-axis ticks
            plt.yticks([])  # Remove y-axis ticks
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, criterion, a, b):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # Accumulated loss
    accu_num = torch.zeros(1).to(device)   # Accumulated number of correctly predicted samples
    optimizer['softmax'].zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, (x,  labels) in enumerate(data_loader):
        labels=labels.to(device)
        sample_num += x.shape[0]
        pred, feat, pred_teacher, feat_teacher = model(x.to(device))
        l_softmax = criterion['softmax'](pred[3], labels)
        for index in range(0, len(feat)-1):
            l_softmax += torch.dist(feat[len(feat)-1-index-1], feat[len(feat)-1-index].detach()) * a
            l_softmax += CrossEntropy(pred[len(feat)-1-index-1], pred[len(feat)-1-index].detach()) * b
        loss = l_softmax
        accu_num += (pred[3].argmax(1) == labels.to(device)).sum()
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.4f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer['softmax'].step()
        optimizer['softmax'].zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch, criterion):

    model.eval()

    accu_num = torch.zeros(1).to(device)   # Accumulated number of correctly predicted samples
    accu_loss = torch.zeros(1).to(device)  # Accumulated loss

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, (x,  labels) in enumerate(data_loader):
        sample_num += x.shape[0]

        pred,_,_,_ = model(x.to(device))
        # pred_teacher = sum(pred[:]) / len(pred)
        pred_classes = torch.max(pred[3], dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        l_softmax = criterion['softmax'](pred[3], labels.to(device))
        loss = l_softmax
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.4f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num,
                                                                                                 )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
