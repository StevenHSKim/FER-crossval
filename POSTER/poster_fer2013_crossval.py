import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import os
import argparse
import cv2
import pandas as pd
import random
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import ShuffleSplit, train_test_split
from time import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

# from data_preprocessing.plot_confusion_matrix import plot_confusion_matrix
from utils import *
from models.emotion_hyp import pyramid_trans_expr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fer2013', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    
    # POSTER model type
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    # Confusion matrix type
    parser.add_argument('-p', '--plot_cm', action="store_true", help="Plotting confusion matrix.")
    
    # batch size
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation/testing.')
    
    # optimizer
    parser.add_argument('--optimizer', type=str, default="adamw", help='Optimizer, adam or adamw.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for adamw.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer')
    
    # train test split ratio
    parser.add_argument('--test_size', default=0.2, type=float, help='Fraction of data to use for testing')
    parser.add_argument('--val_size', default=0.25, type=float, help='Fraction of data to use for validation')
    
    # learning rate scheduler
    parser.add_argument('--lr_scheduler', default='cosine', type=str, choices=['cosine', 'step', 'exp'], help='Learning rate scheduler type')
    parser.add_argument('--t_max', default=20, type=int, help='T_max for CosineAnnealingLR scheduler')
    
    # iterations
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for repeated random sampling')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience.')
    
    return parser.parse_args()

def control_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class FER2013Dataset(data.Dataset):
    def __init__(self, pixels, labels, transform=None, basic_aug=False):
        self.pixels = pixels
        self.labels = labels
        self.transform = transform
        self.basic_aug = basic_aug
        self.aug_func = [flip_image, add_gaussian_noise]

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        pixels = self.pixels[idx].reshape(48, 48)
        image = pixels.astype(np.uint8)
        # Convert to 3 channel by stacking
        image = np.stack([image] * 3, axis=-1)
        image = cv2.resize(image, (224, 224))  # Resize to match RAF-DB size
        
        label = self.labels[idx]

        if self.basic_aug and random.uniform(0, 1) > 0.5:
            index = random.randint(0, 1)
            image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image.copy())

        return image, label

def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def get_data_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_datasets(args, train_val_indices, test_indices, random_state):
    datapath = '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FER2013/fer2013_modified.csv'
    df = pd.read_csv(datapath)
    
    pixels = np.array([np.array([int(p) for p in pixel.split()]) for pixel in df['pixels']])
    labels = df['emotion'].values

    train_indices, val_indices = train_test_split(train_val_indices, test_size=args.val_size, random_state=random_state)

    train_dataset = FER2013Dataset(pixels[train_indices], labels[train_indices], 
                                transform=get_data_transforms(train=True), basic_aug=True)
    val_dataset = FER2013Dataset(pixels[val_indices], labels[val_indices], 
                              transform=get_data_transforms(train=False))
    test_dataset = FER2013Dataset(pixels[test_indices], labels[test_indices], 
                               transform=get_data_transforms(train=False))

    num_classes = 7
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    
    return train_dataset, val_dataset, test_dataset, num_classes, train_size, val_size, test_size

def train(train_loader, val_loader, model, criterion, optimizer, scheduler, device, epochs, patience, iteration):
    best_acc = 0
    best_loss = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    save_path = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, correct_sum, iter_cnt = 0.0, 0, 0
        start_time = time()

        for imgs, targets in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()
            imgs, targets = imgs.to(device), targets.to(device)
            outputs, features = model(imgs)
            CE_loss = criterion['CE'](outputs, targets)
            lsce_loss = criterion['lsce'](outputs, targets)
            loss = 2 * lsce_loss + CE_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicts = torch.max(outputs, 1)
            correct_sum += torch.eq(predicts, targets).sum().item()

        train_acc = correct_sum / len(train_loader.dataset)
        train_loss /= iter_cnt
        elapsed = (time() - start_time) / 60

        print(f'[Epoch {epoch}] Train time: {elapsed:.2f}, Training accuracy: {train_acc:.4f}, Loss: {train_loss:.3f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        scheduler.step()

        model.eval()
        val_loss, bingo_cnt, pre_labels, gt_labels = 0.0, 0, [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                outputs, features = model(imgs.to(device))
                targets = targets.to(device)
                CE_loss = criterion['CE'](outputs, targets)
                val_loss += CE_loss.item()
                _, predicts = torch.max(outputs, 1)
                bingo_cnt += torch.eq(predicts, targets).sum().cpu().item()
                pre_labels += predicts.cpu().tolist()
                gt_labels += targets.cpu().tolist()

        val_loss /= len(val_loader)
        val_acc = bingo_cnt / len(val_loader.dataset)
        f1 = f1_score(pre_labels, gt_labels, average='macro')
        total_score = 0.67 * f1 + 0.33 * val_acc

        print(f"[Epoch {epoch}] Validation accuracy: {val_acc:.4f}, Loss: {val_loss:.3f}, F1 score: {f1:.4f}, Total score: {total_score:.4f}")

        if best_loss >= val_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

        if val_acc > best_acc:
            best_acc = val_acc
            best_val_loss = val_loss
            print(f"Best accuracy: {best_acc:.4f}, Best val loss: {best_val_loss:.4f}")
            
            save_path = os.path.join('/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/POSTER/checkpoint', f"fer2013_iter{iteration+1}_epoch{epoch}_acc{val_acc:.4f}.pth")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)
            print('Model saved.')

    return model, best_acc, best_val_loss, save_path

def test(test_loader, model, checkpoint_path, device, criterion):
    if checkpoint_path is not None:
        print("Loading pretrained weights...", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pre_labels, gt_labels, bingo_cnt = [], [], 0
    running_loss = 0.0
    iter_cnt = 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            outputs, features = model(imgs.to(device))
            targets = targets.to(device)
            CE_loss = criterion['CE'](outputs, targets)
            running_loss += CE_loss.item()
            _, predicts = torch.max(outputs, 1)
            bingo_cnt += torch.eq(predicts, targets).sum().cpu().item()
            pre_labels += predicts.cpu().tolist()
            gt_labels += targets.cpu().tolist()
            iter_cnt += 1

    test_acc = bingo_cnt / len(test_loader.dataset)
    test_acc = np.around(test_acc, 4)
    test_running_loss = running_loss / iter_cnt
    balanced_acc = balanced_accuracy_score(gt_labels, pre_labels)
    balanced_acc = np.around(balanced_acc, 4)
    print(f"Test accuracy: {test_acc:.4f}, Balanced Accuracy: {balanced_acc:.4f}, Running Loss: {test_running_loss:.3f}")
    cm = confusion_matrix(gt_labels, pre_labels)

    return test_acc, balanced_acc, test_running_loss, cm, pre_labels, gt_labels

def run_train_test():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_accuracies = []
    all_test_losses = []
    all_val_losses = []
    best_accuracies = []
    results = []

    # Load all data
    datapath = '/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FER2013/fer2013_modified.csv'
    df = pd.read_csv(datapath)
    
    pixels = np.array([np.array([int(p) for p in pixel.split()]) for pixel in df['pixels']])
    labels = df['emotion'].values

    ss = ShuffleSplit(n_splits=args.iterations, test_size=args.test_size, random_state=42)
    splits = list(ss.split(pixels, labels))

    for iteration, (train_val_indices, test_indices) in enumerate(splits):
        print(f"Iteration {iteration + 1}/{args.iterations}")
        random_state = iteration
        control_random_seed(random_state)

        train_dataset, val_dataset, test_dataset, num_classes, train_size, val_size, test_size = get_datasets(args, train_val_indices, test_indices, random_state)

        print(f"Train set size: {train_size}, Validation set size: {val_size}, Test set size: {test_size}")

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)
        model = torch.nn.DataParallel(model).cuda()
        
        # 모델 파라미터 수 출력
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Total Parameters: %.3fM' % parameters)

        if args.checkpoint:
            print("Loading pretrained weights...", args.checkpoint)
            checkpoint = torch.load(args.checkpoint)
            checkpoint = checkpoint["model_state_dict"]
            model = load_pretrained_weights(model, checkpoint)

        if args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            if args.lr_scheduler == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max)
            elif args.lr_scheduler == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            elif args.lr_scheduler == 'exp':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        else:
            raise ValueError("Optimizer not supported.")

        criterion = {
            'CE': torch.nn.CrossEntropyLoss(),
            'lsce': LabelSmoothingCrossEntropy(smoothing=0.2)
        }

        model, best_acc, best_val_loss, best_checkpoint_path = train(train_loader, val_loader, model, criterion, optimizer, scheduler, device, args.epochs, args.patience, iteration)
        test_acc, test_balanced_acc, test_running_loss, cm, pre_labels, gt_labels = test(test_loader, model, best_checkpoint_path, device, criterion)

        all_accuracies.append(test_acc)
        all_test_losses.append(test_running_loss)
        all_val_losses.append(best_val_loss)
        best_accuracies.append(best_acc)
        results.append([iteration + 1, test_acc, test_balanced_acc, best_val_loss, test_running_loss])
        print(f"Test Accuracy: {test_acc}, Test Balanced Accuracy: {test_balanced_acc}, Val Loss: {best_val_loss:.4f}, Test Loss: {test_running_loss:.4f}")

        if args.plot_cm:
            labels_name = ['AN', 'DI', 'FE', 'HA', 'SA', 'SU', "NE"]
            plot_confusion_matrix(cm, labels_name, f'FER2013 Iteration {iteration + 1}', test_acc)

    # Calculate metrics with mean and std
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_val_loss = np.mean(all_val_losses)
    std_val_loss = np.std(all_val_losses)
    mean_test_loss = np.mean(all_test_losses)
    std_test_loss = np.std(all_test_losses)

    # Save results to POSTER_FER2013_test_results.csv
    results_df = pd.DataFrame(results, columns=['Iteration', 'Test Accuracy', 'Balanced Accuracy', 'Val Loss', 'Test Loss'])
    
    # Add summary row
    summary = pd.DataFrame([{
        'Iteration': 'Mean±Std',
        'Test Accuracy': f"{mean_accuracy:.4f}±{std_accuracy:.4f}",
        'Balanced Accuracy': "-",
        'Val Loss': f"{mean_val_loss:.4f}±{std_val_loss:.4f}",
        'Test Loss': f"{mean_test_loss:.4f}±{std_test_loss:.4f}"
    }])
    results_df = pd.concat([results_df, summary])
    results_df.to_csv('POSTER_FER2013_test_results.csv', index=False)
    
    print(f"\nResults over {args.iterations} iterations:")
    print(f"Test Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"Test Loss: {mean_test_loss:.4f} ± {std_test_loss:.4f}")
    
    # Get current time for experiment logging
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
    
    # Prepare the new result row for total_results.csv
    new_result = {
        'Experiment Time': current_time,
        'Train Time': current_time,
        'Iteration': args.iterations,
        'Dataset Name': 'FER2013',
        'Data Split': f'{int((1-args.test_size)*(1-args.val_size)*100)}:{int((1-args.test_size)*args.val_size*100)}:{int(args.test_size*100)}',
        'Model Name': f'POSTER_{args.modeltype}',
        'Total Parameters': f"{parameters:.3f}M",
        'Val Loss': f"{mean_val_loss:.4f}±{std_val_loss:.4f}",
        'Test Loss': f"{mean_test_loss:.4f}±{std_test_loss:.4f}",
        'Acc': f"{mean_accuracy:.4f}±{std_accuracy:.4f}"
    }
    
    # Define expected columns for total_results.csv
    expected_columns = ['Experiment Time', 'Train Time', 'Iteration', 'Dataset Name', 'Data Split', 
                       'Model Name', 'Total Parameters', 'Val Loss', 'Test Loss', 'Acc']
    
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

if __name__ == "__main__":
    run_train_test()