import os
import sys
import warnings
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ShuffleSplit, train_test_split
from networks.dan import DAN
import random
from datetime import datetime

def warn(*args, **kwargs):
    pass
warnings.warn = warn

eps = sys.float_info.epsilon

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fer_path', type=str, default='/userHome/userhome1/automl_undergraduate/FER_Models/datasets/FER2013/fer2013_modified.csv', help='FER2013 dataset path.')
    parser.add_argument('--gpu', type=str, default='0', help='Assign a single GPU by its number.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    
    # DAN
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    
    # batch size
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.') # 128
    
    # optimizer
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for AdamW.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    
    # train test split ratio
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.25, help='Fraction of data to use for validation')
    
    # learning rate scheduler
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'exp'], help='Learning rate scheduler type')
    parser.add_argument('--t_max', type=int, default=20, help='T_max for CosineAnnealingLR scheduler')
    
    # iterations
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for repeated random sampling')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience.')
    
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
    def __init__(self, fer_path, indices, transform=None):
        self.transform = transform
        
        # Read CSV file
        df = pd.read_csv(fer_path)
        self.pixels = df['pixels'].values[indices]
        self.labels = df['emotion'].values[indices]
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pixels = self.pixels[idx].split()
        pixels = np.array([int(pixel) for pixel in pixels], dtype=np.uint8)
        image = pixels.reshape(48, 48)
        image = Image.fromarray(image)
        
        # Convert grayscale to RGB
        image = image.convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=7, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))
        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

class PartitionLoss(nn.Module):
    def __init__(self):
        super(PartitionLoss, self).__init__()

    def forward(self, x):
        num_head = x.size(1)
        if num_head > 1:
            var = x.var(dim=1).mean()
            loss = torch.log(1 + num_head / (var + eps))
        else:
            loss = 0
        return loss

def train(train_loader, val_loader, model, criterion_cls, criterion_af, criterion_pt, optimizer, scheduler, device, epochs, patience, iteration):
    best_loss = float('inf')
    best_acc = 0
    best_checkpoint_path = None
    patience_counter = 0

    for epoch in tqdm(range(1, epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.to(device)
            targets = targets.to(device)
            out, feat, heads = model(imgs)

            loss = criterion_cls(out, targets) + criterion_af(feat, targets) + criterion_pt(heads)
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(len(train_loader.dataset))
        running_loss = running_loss / iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss, optimizer.param_groups[0]['lr']))
        
        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            
            y_true = []
            y_pred = []

            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                out, feat, heads = model(imgs)
                loss = criterion_cls(out, targets) + criterion_af(feat, targets) + criterion_pt(heads)

                val_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                
                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())
        
            val_loss = val_loss / iter_cnt   
            scheduler.step()

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            best_acc = max(acc, best_acc)
            
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

            current_time = datetime.now().strftime('%y%m%d_%H%M%S')
            tqdm.write("[%s] [Epoch %d] Validation accuracy: %.4f. bacc: %.4f. Loss: %.3f" % (
                current_time, epoch, acc, balanced_acc, val_loss))
            tqdm.write("best_acc:" + str(best_acc))
            
            if acc > 0.65 and acc == best_acc:
                best_checkpoint_path = os.path.join('/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/DAN/checkpoints',
                                                  f"fer2013_iter{iteration+1}_epoch{epoch}_acc{acc}_bacc{balanced_acc}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, best_checkpoint_path)
                tqdm.write('Model saved.')

            if best_loss >= val_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return best_checkpoint_path, best_acc, best_loss


def test(test_loader, model, checkpoint_path, criterion_cls, criterion_af, criterion_pt, device):
    if checkpoint_path:  # checkpoint가 있는 경우에만 로드
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0

        y_true = []
        y_pred = []

        model.eval()
        for (imgs, targets) in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            out, feat, heads = model(imgs)
            loss = criterion_cls(out, targets) + criterion_af(feat, targets) + criterion_pt(heads)

            running_loss += loss
            iter_cnt += 1
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += out.size(0)

            y_true.append(targets.cpu().numpy())
            y_pred.append(predicts.cpu().numpy())

        running_loss = running_loss / iter_cnt

        acc = bingo_cnt.float() / float(sample_cnt)
        acc = np.around(acc.numpy(), 4)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

        current_time = datetime.now().strftime('%y%m%d_%H%M%S')
        tqdm.write("[%s] Test accuracy: %.4f. bacc: %.4f. Loss: %.3f" % (
            current_time, acc, balanced_acc, running_loss))

        return acc, balanced_acc, running_loss.item()

def run_train_test():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_accuracies = []
    best_accuracies = []
    all_val_losses = []
    all_test_losses = []
    results = []

    df = pd.read_csv(args.fer_path)
    indices = np.arange(len(df))
    labels = df['emotion'].values

    ss = ShuffleSplit(n_splits=args.iterations, test_size=args.test_size, random_state=42)
    splits = list(ss.split(indices, labels))

    for iteration, (train_val_indices, test_indices) in enumerate(splits):
        print(f"Iteration {iteration + 1}/{args.iterations}")
        control_random_seed(iteration)

        model = DAN(num_head=args.num_head, num_class=7)  # FER2013 has 7 emotion classes
        model.to(device)
        
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Total Parameters: %.3fM' % parameters)

        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32)
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.25)),
        ])

        train_indices, val_indices = train_test_split(train_val_indices, test_size=args.val_size, random_state=iteration)

        train_dataset = FER2013Dataset(args.fer_path, indices=train_indices, transform=data_transforms)
        print('Train set size:', train_dataset.__len__())

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.workers,
                                                 shuffle=True,
                                                 pin_memory=True)

        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = FER2013Dataset(args.fer_path, indices=val_indices, transform=val_transforms)
        print('Validation set size:', val_dataset.__len__())

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)

        test_dataset = FER2013Dataset(args.fer_path, indices=test_indices, transform=val_transforms)
        print('Test set size:', test_dataset.__len__())

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch_size,
                                                num_workers=args.workers,
                                                shuffle=False,
                                                pin_memory=True)

        criterion_cls = torch.nn.CrossEntropyLoss()
        criterion_af = AffinityLoss(device, num_class=7)  # FER2013 has 7 emotion classes
        criterion_pt = PartitionLoss()

        params = list(model.parameters()) + list(criterion_af.parameters())
        
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

        if args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
        elif args.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif args.lr_scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        best_checkpoint_path, best_acc, best_val_loss = train(train_loader, val_loader, model, criterion_cls, criterion_af, criterion_pt, 
                                             optimizer, scheduler, device, args.epochs, args.early_stopping_patience, iteration)

        if best_checkpoint_path:
            test_acc, test_balanced_acc, test_running_loss = test(test_loader, model, best_checkpoint_path, 
                                                                criterion_cls, criterion_af, criterion_pt, device)
            all_accuracies.append(test_acc)
            best_accuracies.append(best_acc)
            all_val_losses.append(best_val_loss)
            all_test_losses.append(test_running_loss)
            results.append([iteration + 1, test_acc, test_balanced_acc, test_running_loss, best_val_loss])
            print(f"Test Accuracy: {test_acc}, Test Balanced Accuracy: {test_balanced_acc}")
            print(f"Test Loss: {test_running_loss:.4f}, Best Val Loss: {best_val_loss:.4f}")

    results_df = pd.DataFrame(results, columns=['Iteration', 'Test Accuracy', 'Balanced Accuracy', 'Test Loss', 'Val Loss'])
    results_df.to_csv('DAN_FER2013_test_results.csv', index=False)

    print("\nBest Accuracies over all iterations:")
    for i, acc in enumerate(best_accuracies, 1):
        print(f"Iteration {i}: {acc:.4f}")

    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_val_loss = np.mean(all_val_losses)
    std_val_loss = np.std(all_val_losses)
    mean_test_loss = np.mean(all_test_losses)
    std_test_loss = np.std(all_test_losses)

    print(f"\nResults over {args.iterations} iterations:")
    print(f"Test Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"Test Loss: {mean_test_loss:.4f} ± {std_test_loss:.4f}")

    # Get current time for experiment logging
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')

    # Get total parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    
    # Prepare the new result row
    new_result = {
        'Experiment Time': current_time,
        'Train Time': current_time,
        'Iteration': args.iterations,
        'Dataset Name': 'FER2013',
        'Data Split': f'{int((1-args.test_size)*(1-args.val_size)*100)}:{int(args.test_size*100)}:{int((1-args.test_size)*args.val_size*100)}',
        'Model Name': 'DAN',
        'Val Loss': f"{mean_val_loss:.4f}±{std_val_loss:.4f}",
        'Test Loss': f"{mean_test_loss:.4f}±{std_test_loss:.4f}",
        'Acc': f"{mean_accuracy:.4f}±{std_accuracy:.4f}",
        'Total Parameters': f"{total_params:.3f}M"
    }

    # Define expected columns
    expected_columns = ['Experiment Time', 'Train Time', 'Iteration', 'Dataset Name', 
                       'Data Split', 'Model Name', 'Val Loss', 'Test Loss', 'Acc', 'Total Parameters']

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