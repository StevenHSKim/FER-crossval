import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime

from models.PosterV2_7cls import pyramid_trans_expr2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/userHome/userhome3/timur/Haesung/datasets/FER2013/fer2013_modified.csv')
    parser.add_argument('--model_name', type=str, default='POSTER++')
    parser.add_argument('--dataset', type=str, default='fer2013')
    
    parser.add_argument('--gpu', type=str, default='0,1')
    parser.add_argument('--workers', default=8, type=int)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.25)
    
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'exp'])
    parser.add_argument('--t_max', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)
    return parser.parse_args()

class FER2013Dataset(data.Dataset):
    def __init__(self, pixels, labels, transform=None):
        self.pixels = pixels
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pixels = self.pixels[idx].reshape(48, 48)
        image = Image.fromarray(pixels.astype('uint8'), 'L').convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, label

def control_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(train_loader, val_loader, model, criterion, optimizer, scheduler, device, epochs, patience, iteration, dataset_name, checkpoint_dir):
    scaler = torch.cuda.amp.GradScaler()
    accumulation_steps = 4
    best_acc = 0
    best_loss = float('inf')
    patience_counter = 0
    best_checkpoint_path = None
    final_val_loss = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        optimizer.zero_grad()

        for i, (images, targets) in enumerate(train_loader):
            iter_cnt += 1
            images, targets = images.to(device), targets.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if i % 10 == 0:
                torch.cuda.empty_cache()
            
            running_loss += loss.item() * accumulation_steps
            _, predicts = torch.max(outputs, 1)
            correct_sum += torch.eq(predicts, targets).sum()

        train_acc = correct_sum.float() / len(train_loader.dataset)
        # train_loss = running_loss / iter_cnt
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_iter_cnt = 0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicts = torch.max(outputs, 1)
                val_correct += torch.eq(predicts, targets).sum()
                
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicts.cpu().numpy())
                val_iter_cnt += 1
        
        val_acc = val_correct.float() / len(val_loader.dataset)
        val_loss = val_loss / val_iter_cnt
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        final_val_loss = val_loss
        
        print(f'Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, B-Acc: {balanced_acc:.4f}')
        
        scheduler.step()

        if val_loss < best_loss:
            best_acc = val_acc
            previous_checkpoint = None
            for filename in os.listdir(checkpoint_dir):
                if filename.startswith(f"posterv2_{dataset_name}_iter{iteration+1}_epoch") and filename.endswith(".pth"):
                    previous_checkpoint = os.path.join(checkpoint_dir, filename)
                    break

            if previous_checkpoint and os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)

            best_checkpoint_path = os.path.join(
                checkpoint_dir,
                f"posterv2_{dataset_name}_iter{iteration+1}_epoch{epoch}_acc{val_acc:.4f}_bacc{balanced_acc:.4f}.pth",
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_checkpoint_path)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    return best_checkpoint_path, best_acc, final_val_loss

def test(test_loader, model, checkpoint_path, criterion, device):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loss = 0.0
    correct = 0
    iter_cnt = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicts = torch.max(outputs, 1)
            correct += torch.eq(predicts, targets).sum()
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicts.cpu().numpy())
            iter_cnt += 1

    test_acc = correct.float() / len(test_loader.dataset)
    test_loss = test_loss / iter_cnt
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return test_acc.item(), balanced_acc, test_loss

def run_train_test():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device_ids = list(range(len(args.gpu.split(','))))
    device = torch.device(f'cuda:{device_ids[0]}')
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.7, i)
            
    model = pyramid_trans_expr2(img_size=48, num_classes=7)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    del model
    
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
    checkpoint_dir = '/userHome/userhome3/timur/Haesung/POSTER++/checkpoint/fer2013'
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_accuracies = []
    all_balanced_accuracies = []
    all_val_losses = []
    all_test_losses = []
    results = []

    img_size = 48
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load FER2013 dataset
    df = pd.read_csv(args.data)
    pixels = np.array([np.array([int(pixel) for pixel in pixels.split()]) for pixels in df['pixels']])
    labels = df['emotion'].values

    ss = ShuffleSplit(n_splits=args.iterations, test_size=args.test_size, random_state=42)
    splits = list(ss.split(pixels, labels))

    for iteration, (train_val_indices, test_indices) in enumerate(splits):
        print(f"\nIteration {iteration + 1}/{args.iterations}")
        control_random_seed(iteration)

        train_indices, val_indices = train_test_split(train_val_indices, test_size=args.val_size, random_state=iteration)

        train_dataset = FER2013Dataset(
            pixels[train_indices],
            labels[train_indices],
            transform=train_transform
        )
        val_dataset = FER2013Dataset(
            pixels[val_indices],
            labels[val_indices],
            transform=val_transform
        )
        test_dataset = FER2013Dataset(
            pixels[test_indices],
            labels[test_indices],
            transform=val_transform
        )

        print(f'Train set size: {len(train_dataset)}')
        print(f'Validation set size: {len(val_dataset)}')
        print(f'Test set size: {len(test_dataset)}')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        model = pyramid_trans_expr2(img_size=img_size, num_classes=7)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        if args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
        if args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
        elif args.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        best_checkpoint_path, best_acc, val_loss = train(
            train_loader, val_loader, model, criterion, 
            optimizer, scheduler, device, args.epochs, args.patience,
            iteration, args.dataset, checkpoint_dir
        )

        if best_checkpoint_path:
            test_acc, test_balanced_acc, test_loss = test(
                test_loader, model, best_checkpoint_path, criterion, device
            )
            
            all_accuracies.append(test_acc)
            all_balanced_accuracies.append(test_balanced_acc)
            all_val_losses.append(val_loss)
            all_test_losses.append(test_loss)
            results.append([iteration + 1, test_acc, test_balanced_acc, val_loss, test_loss])
            print(f"Test Accuracy: {test_acc:.4f}, Balanced Accuracy: {test_balanced_acc:.4f}, Loss: {test_loss:.4f}")

    metrics = {
        'accuracy': (np.mean(all_accuracies), np.std(all_accuracies)),
        'balanced_accuracy': (np.mean(all_balanced_accuracies), np.std(all_balanced_accuracies)),
        'val_loss': (np.mean(all_val_losses), np.std(all_val_losses)),
        'test_loss': (np.mean(all_test_losses), np.std(all_test_losses))
    }

    results_df = pd.DataFrame(results, columns=['Iteration', 'Test Accuracy', 'Balanced Accuracy', 'Val Loss', 'Test Loss'])
    summary = pd.DataFrame([{
        'Iteration': args.iterations,
        'Test Accuracy': f"{metrics['accuracy'][0]:.4f}±{metrics['accuracy'][1]:.4f}",
        'Balanced Accuracy': f"{metrics['balanced_accuracy'][0]:.4f}±{metrics['balanced_accuracy'][1]:.4f}",
        'Val Loss': f"{metrics['val_loss'][0]:.4f}±{metrics['val_loss'][1]:.4f}",
        'Test Loss': f"{metrics['test_loss'][0]:.4f}±{metrics['test_loss'][1]:.4f}",
    }])
    
    results_df = pd.concat([results_df, summary])
    results_df.to_csv(f'{args.model_name}_{args.dataset}_test_results.csv', index=False)

    new_result = {
        'Experiment Time': current_time,
        'Train Time': current_time,
        'Iteration': args.iterations,
        'Dataset Name': args.dataset.upper(),
        'Data Split': f'{int((1-args.test_size)*(1-args.val_size)*100)}:{int((1-args.test_size)*args.val_size*100)}:{int(args.test_size*100)}',
        'Model Name': args.model_name,
        'Total Parameters': f"{parameters:.3f}M",
        'Val Loss': f"{metrics['val_loss'][0]:.4f}±{metrics['val_loss'][1]:.4f}",
        'Test Loss': f"{metrics['test_loss'][0]:.4f}±{metrics['test_loss'][1]:.4f}",
        'Acc': f"{metrics['accuracy'][0]:.4f}±{metrics['accuracy'][1]:.4f}",
        'Balanced_Acc': f"{metrics['balanced_accuracy'][0]:.4f}±{metrics['balanced_accuracy'][1]:.4f}"
    }
    
    expected_columns = ['Experiment Time', 'Train Time', 'Iteration', 'Dataset Name', 'Data Split', 'Model Name', 'Total Parameters', 'Val Loss', 'Test Loss', 'Acc', 'Balanced_Acc']

    results_path = '/userHome/userhome3/timur/Haesung/total_results.csv'
    os.makedirs(results_path, exist_ok=True)
    
    try:
        total_results_df = pd.read_csv(results_path)
        if total_results_df.empty or not all(col in total_results_df.columns for col in expected_columns):
            total_results_df = pd.DataFrame(columns=expected_columns)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        total_results_df = pd.DataFrame(columns=expected_columns)
    
    total_results_df = pd.concat([total_results_df, pd.DataFrame([new_result])], ignore_index=True)
    total_results_df.to_csv(results_path, index=False)
    print(f"\nResults have been appended to {results_path}")

if __name__ == "__main__":
    run_train_test()