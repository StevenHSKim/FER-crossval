import argparse
import math
import time
import os
import random
from datetime import datetime
import numpy as np
import pandas as pd
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import ShuffleSplit, train_test_split
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from PIL import Image
import torchvision.transforms as transforms

from model.model import create_model
from utils import Logger, AverageMeter, generate_adaptive_LD, generate_average_weights, get_accuracy
from auto_augment import rand_augment_transform


# Argument parser
parser = argparse.ArgumentParser(description='PyTorch Training')
# Training configurations
parser.add_argument('--epochs', default=100, type=int)          
parser.add_argument('--batch_size', default=64, type=int)      
parser.add_argument('--lr', default=0.001, type=float)          
parser.add_argument('--optimizer', default='AdamW', type=str)   
parser.add_argument('--weight_decay', default=1e-4, type=float) 
parser.add_argument('--num_classes', default=8, type=int)       # FERPlus has 8 classes
# Method-specific configurations
parser.add_argument('--threshold', default=0.7, type=float)
parser.add_argument('--sharpen', default=False, type=bool)
parser.add_argument('--T', default=1.2, type=float)
parser.add_argument('--alpha', default=None, type=float)
parser.add_argument('--beta', default=3, type=int)
parser.add_argument('--max_weight', default=1.0, type=float)
parser.add_argument('--min_weight', default=0.2, type=float)
parser.add_argument('--drop_rate', default=0.0, type=float)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--label_smoothing', default=0.0, type=float)
parser.add_argument('--tops', default=0.7, type=float)
parser.add_argument('--margin_1', default=0.07, type=float)
# Common configurations
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--dataset', default='ferplus', type=str)
parser.add_argument('--data_path', default='/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FERPlus/FERPlus_Label_modified.csv', type=str)
parser.add_argument('--image_dir', default='/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FERPlus/FERPlus_Image', type=str)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--iterations', default=10, type=int)
parser.add_argument('--test_size', default=0.2, type=float)
parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience.')

args = parser.parse_args()

best_acc = 0
best_epoch = 0

# Set device
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

def control_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set random seed if provided
if args.seed is not None:
    control_random_seed(args.seed)
    

# FERPlus Dataset Class
class FERPlusDataset(Dataset):
    def __init__(self, data_path, image_dir, transform=None):
        self.transform = transform
        self.image_dir = image_dir
        
        # Load the CSV file
        df = pd.read_csv(data_path)
        self.image_names = df['Image name'].values
        self.label = df['label'].values
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Load image file
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        # Read as grayscale image
        image = Image.open(image_path)
        # Convert to RGB (3 channels)
        image = image.convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, self.label[idx], idx

def get_ferplus_dataloaders(data_path, image_dir, batch_size=64, num_workers=2, 
                         train_indices=None, val_indices=None, test_indices=None):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        rand_augment_transform(config_str='rand-m5-n3-mstd0.5', hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])
    
    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Create full dataset with training transforms
    full_dataset = FERPlusDataset(data_path, image_dir, transform=data_transforms)

    # Create subsets using indices
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    
    # Create validation and test datasets with validation transforms
    full_dataset_val = FERPlusDataset(data_path, image_dir, transform=data_transforms_val)
    val_dataset = torch.utils.data.Subset(full_dataset_val, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset_val, test_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True, 
        drop_last=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader

def save_checkpoint(state, is_best, iteration, checkpoint='/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/Ada-DF/checkpoints', filename='checkpoint.pth'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    
    filepath = os.path.join(checkpoint, f'ferplus_checkpoint_iter{iteration}.pth')
    torch.save(state, filepath)
    
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f'ferplus_model_best_iter{iteration}.pth'))


def train(train_loader, model, criterion, criterion_kld, optimizer, LD, epoch):
    if args.alpha is not None:
        alpha_1 = args.alpha
        alpha_2 = 1 - args.alpha
    else:
        if epoch <= args.beta:
            alpha_1 = math.exp(-(1 - epoch / args.beta) ** 2)
            alpha_2 = 1
        else:
            alpha_1 = 1
            alpha_2 = math.exp(-(1 - args.beta / epoch) ** 2)

    losses = AverageMeter()
    losses_ce = AverageMeter()
    losses_kld = AverageMeter()
    losses_rr = AverageMeter()
    accs = AverageMeter()

    model.train()

    outputs_list = []
    targets_list = []
    weights_list = []

    for i, (images, labels, idxs) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        idxs = idxs.to(device)

        outputs_1, outputs_2, attention_weights = model(images)

        batch_size = images.size(0)
        tops = int(batch_size * args.tops)
        _, top_idx = torch.topk(attention_weights.squeeze(), tops)
        _, down_idx = torch.topk(attention_weights.squeeze(), batch_size - tops, largest=False)

        high_group = attention_weights[top_idx]
        low_group = attention_weights[down_idx]
        high_mean = torch.mean(high_group)
        low_mean = torch.mean(low_group)
        diff = low_mean - high_mean + args.margin_1

        if diff > 0:
            RR_loss = diff
        else:
            RR_loss = torch.tensor(0.0).to(device)

        loss_ce = criterion(outputs_1, labels).mean()

        attention_weights = attention_weights.squeeze(1)
        attention_weights = ((attention_weights - attention_weights.min()) /
                             (attention_weights.max() - attention_weights.min())) * \
                            (args.max_weight - args.min_weight) + args.min_weight
        attention_weights = attention_weights.unsqueeze(1)

        labels_onehot = F.one_hot(labels, args.num_classes)
        targets = (1 - attention_weights) * F.softmax(outputs_1, dim=1) + \
                  attention_weights * LD[labels]

        loss_kld = criterion_kld(F.log_softmax(outputs_2, dim=1), targets).sum() / batch_size

        loss = alpha_2 * loss_ce + alpha_1 * loss_kld + RR_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs_list.append(outputs_1.detach())
        targets_list.append(labels.detach())
        weights_list.append(attention_weights.detach())

        top1 = get_accuracy(outputs_1, labels, topk=(1,))[0].item()
        losses.update(loss.item(), images.size(0))
        losses_ce.update(loss_ce.item(), images.size(0))
        losses_kld.update(loss_kld.item(), images.size(0))
        losses_rr.update(RR_loss.item(), images.size(0))
        accs.update(top1, images.size(0))

    outputs_new = torch.cat(outputs_list, dim=0)
    targets_new = torch.cat(targets_list, dim=0)
    weights_new = torch.cat(weights_list, dim=0)

    return losses.avg, losses_ce.avg, losses_kld.avg, alpha_1, alpha_2, accs.avg, outputs_new, targets_new, weights_new

def validate(loader, model, criterion, epoch, phase='train'):
    losses = AverageMeter()
    accs = AverageMeter()
    all_targets = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            if phase == 'train':
                outputs, _, _ = model(images)
            else:
                _, outputs, _ = model(images)

            loss = criterion(outputs, labels).mean()

            top1 = get_accuracy(outputs, labels, topk=(1,))[0].item()
            losses.update(loss.item(), images.size(0))
            accs.update(top1, images.size(0))

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100

    return losses.avg, accs.avg, balanced_acc

def main():
    global device

    logger = Logger('./results/log-' + time.strftime('%b%d_%H-%M-%S') + '.txt')
    logger.info(args)
    writer = SummaryWriter()

    # Dataset loading
    df = pd.read_csv(args.data_path)
    file_names = df['Image name'].values
    labels = df['label'].values
    
    print(f"\nTotal dataset size: {len(file_names)}")
    logger.info(f"Total dataset size: {len(file_names)}")

    ss = ShuffleSplit(n_splits=args.iterations, test_size=args.test_size, random_state=42)
    splits = list(ss.split(file_names, labels))

    all_accuracies = []
    all_balanced_accuracies = []
    best_accuracies = []
    all_val_losses = []
    all_test_losses = []
    results = []

    for iteration, (train_val_indices, test_indices) in enumerate(splits):
        print(f"Iteration {iteration + 1}/{args.iterations}")
        logger.info(f"Iteration {iteration + 1}/{args.iterations}")

        best_acc = 0
        best_epoch = 0
        best_loss = float('inf')
        patience_counter = 0
        patience = args.early_stopping_patience

        control_random_seed(iteration)

        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.25, random_state=iteration)
            
        print(f"\nIteration {iteration + 1} Dataset Split:")
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        print(f"Test samples: {len(test_indices)}")

        logger.info(f"Training samples: {len(train_indices)}")
        logger.info(f"Validation samples: {len(val_indices)}")
        logger.info(f"Test samples: {len(test_indices)}")

        train_loader, val_loader, test_loader = get_ferplus_dataloaders(
            args.data_path, args.image_dir, args.batch_size, args.num_workers,
            train_indices, val_indices, test_indices)

        logger.info('Load model...')
        model = create_model(args.num_classes, args.drop_rate).to(device)

        # Initialize LD matrix
        LD = torch.zeros(args.num_classes, args.num_classes).to(device)
        for i in range(args.num_classes):
            LD[i] = torch.zeros(args.num_classes).fill_((1 - args.threshold) / (args.num_classes - 1)).scatter_(0, torch.tensor(i), args.threshold)
        if args.sharpen:
            LD = torch.pow(LD, 1 / args.T) / torch.sum(torch.pow(LD, 1 / args.T), dim=1)
        LD_max = torch.max(LD, dim=1)
        LD_sum = LD

        nan = float('nan')
        weights_avg = [nan for _ in range(args.num_classes)]
        weights_max = [nan for _ in range(args.num_classes)]
        weights_min = [nan for _ in range(args.num_classes)]

        criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=args.label_smoothing)
        criterion_kld = nn.KLDivLoss(reduction='none')

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

        logger.info('Start training.')

        for epoch in range(1, args.epochs + 1):
            logger.info('----------------------------------------------------------')
            logger.info('Epoch: %d, Learning Rate: %f', epoch, optimizer.param_groups[0]['lr'])
            logger.info(f'Maximums of LD: {[round(LD_max[0].cpu().tolist()[i], 4) for i in range(args.num_classes)]}')
            logger.info(f'Average weights: {[round(weights_avg[i], 4) for i in range(args.num_classes)]}')
            logger.info(f'Max weights: {[round(weights_max[i], 4) for i in range(args.num_classes)]}')
            logger.info(f'Min weights: {[round(weights_min[i], 4) for i in range(args.num_classes)]}')

            train_loss, train_loss_ce, train_loss_kld, alpha_1, alpha_2, train_acc, outputs_new, targets_new, weights_new = train(
                train_loader, model, criterion, criterion_kld, optimizer, LD, epoch)

            LD = generate_adaptive_LD(outputs_new, targets_new, args.num_classes, args.threshold, args.sharpen, args.T)
            LD_max = torch.max(LD, dim=1)
            weights_avg, weights_max, weights_min = generate_average_weights(weights_new, targets_new, args.num_classes, args.max_weight, args.min_weight)

            val_loss, val_acc, val_balanced_acc = validate(val_loader, model, criterion, epoch, phase='validation')
            test_loss, test_acc, test_balanced_acc = validate(test_loader, model, criterion, epoch, phase='test')

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                print("Early stopping triggered")
                break

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch

            logger.info('Alpha_1, Alpha_2: %.2f, %.2f Beta: %.2f', alpha_1, alpha_2, args.beta)
            logger.info('Train Loss: %.4f (CE: %.4f, KLD: %.4f)', train_loss, train_loss_ce, train_loss_kld)
            logger.info('Train Acc: %.2f', train_acc)
            logger.info('Val Loss: %.4f, Val Acc: %.2f, Balanced Val Acc: %.2f', val_loss, val_acc, val_balanced_acc)
            logger.info('Test Loss: %.4f, Test Acc: %.2f, Balanced Test Acc: %.2f', test_loss, test_acc, test_balanced_acc)
            logger.info('Best Acc: %.2f (%d)', best_acc, best_epoch)

            is_best = (best_epoch == epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'class_distributions': LD.detach(),
            }, is_best, iteration)

            scheduler.step()

        all_accuracies.append(test_acc)
        all_balanced_accuracies.append(test_balanced_acc)
        best_accuracies.append(best_acc)
        all_val_losses.append(val_loss)
        all_test_losses.append(test_loss)
        results.append([iteration + 1, test_acc, test_balanced_acc, val_loss, test_loss])

    logger.info('All Iteration Test Accuracies: %s', all_accuracies)
    logger.info('All Iteration Balanced Test Accuracies: %s', all_balanced_accuracies)
    logger.info('All Iteration Validation Losses: %s', all_val_losses)
    logger.info('All Iteration Test Losses: %s', all_test_losses)
    logger.info('Best Accuracies Across Iterations: %s', best_accuracies)

    # Save detailed results to CSV
    results_df = pd.DataFrame(
        results, 
        columns=['Iteration', 'Test Accuracy', 'Balanced Accuracy', 'Validation Loss', 'Test Loss']
    )
    results_df.to_csv('AdaDF_FERPlus_test_results.csv', index=False)

    print("\nBest Accuracies over all iterations:")
    for i, acc in enumerate(best_accuracies, 1):
        print(f"Iteration {i}: {acc:.4f}")

    # Calculate means and standard deviations
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_balanced_accuracy = np.mean(all_balanced_accuracies)
    std_balanced_accuracy = np.std(all_balanced_accuracies)
    mean_val_loss = np.mean(all_val_losses)
    std_val_loss = np.std(all_val_losses)
    mean_test_loss = np.mean(all_test_losses)
    std_test_loss = np.std(all_test_losses)

    print(f"\nResults over {args.iterations} iterations:")
    print(f"Test Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Balanced Test Accuracy: {mean_balanced_accuracy:.4f} ± {std_balanced_accuracy:.4f}")
    print(f"Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"Test Loss: {mean_test_loss:.4f} ± {std_test_loss:.4f}")

    # Get current time for experiment logging
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')

    # Calculate total parameters in millions
    parameters = sum(p.numel() for p in model.parameters()) / 1e6

    # Prepare the new result row
    new_result = {
        'Experiment Time': current_time,
        'Train Time': current_time,
        'Iteration': args.iterations,
        'Dataset Name': 'FERPlus',
        'Data Split': f'60:20:20',
        'Model Name': 'Ada-DF',
        'Total Parameters': f"{parameters:.3f}M",
        'Val Loss': f"{mean_val_loss:.4f} ± {std_val_loss:.4f}",
        'Test Loss': f"{mean_test_loss:.4f} ± {std_test_loss:.4f}",
        'Acc': f"{mean_accuracy:.4f} ± {std_accuracy:.4f}"
    }

    # Define expected columns
    expected_columns = ['Experiment Time', 'Train Time', 'Iteration', 'Dataset Name', 
                       'Data Split', 'Model Name', 'Total Parameters', 'Val Loss', 
                       'Test Loss', 'Acc']

    # Read existing CSV file or create new one if it doesn't exist
    results_path = '/userHome/userhome1/automl_undergraduate/FER_Models/total_results.csv'
    try:
        total_results_df = pd.read_csv(results_path)
        # Check if file is empty or missing columns
        if total_results_df.empty or not all(col in total_results_df.columns for col in expected_columns):
            total_results_df = pd.DataFrame(columns=expected_columns)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        total_results_df = pd.DataFrame(columns=expected_columns)

    # Append new result to the DataFrame
    total_results_df = pd.concat([total_results_df, pd.DataFrame([new_result])], ignore_index=True)

    # Save updated results
    total_results_df.to_csv(results_path, index=False)
    print(f"\nResults have been appended to {results_path}")


if __name__ == '__main__':
    main()