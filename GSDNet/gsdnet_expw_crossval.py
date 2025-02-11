import os
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime

from model.gsdnet import GSDNet

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/userHome/userhome1/automl_undergraduate/FER_Models/datasets/ExpW')
    parser.add_argument('--num_class', type=int, default=7)
    parser.add_argument('--img_size', type=int, default=112)
    
    parser.add_argument('--model_name', type=str, default='GSDNet')
    parser.add_argument('--dataset_name', type=str, default='expw')
    
    parser.add_argument('--gpu', type=str, default='0,1')
    parser.add_argument('--workers', default=8, type=int)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    parser.add_argument('--t_max', type=int, default=100)
    
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.25)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)
    
    # GSDNet specific parameters
    parser.add_argument('--a', type=float, default=0.01)
    parser.add_argument('--b', type=float, default=0.3)
    return parser.parse_args()

class ExpWDataset(data.Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label 

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
    softmax_targets = F.softmax(targets/3.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def control_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.deterministic = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def train(train_loader, val_loader, model, criterion, optimizer, device, epoch, a, b,
         checkpoint_dir, dataset_name, model_name, iteration, best_loss, patience_counter, patience):
    # GradScaler 초기화
    scaler = torch.cuda.amp.GradScaler(
        init_scale=2**16,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True
    )
    
    # CUDA 스트림 설정
    train_stream = torch.cuda.Stream()
    
    model.train()
    running_loss = 0.0
    correct_sum = 0
    iter_cnt = 0
    best_checkpoint_path = None
    
    # 그래디언트 누적 설정
    accumulation_steps = 4

    for i, (images, targets) in enumerate(train_loader):
        iter_cnt += 1
        
        # non_blocking 전송으로 데이터 로딩 최적화
        with torch.cuda.stream(train_stream):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            pred, feat, pred_teacher, feat_teacher = model(images)
            
            # Calculate loss
            l_softmax = criterion['softmax'](pred[3], targets)
            for index in range(0, len(feat)-1):
                l_softmax += torch.dist(feat[len(feat)-1-index-1], feat[len(feat)-1-index].detach()) * a
                l_softmax += CrossEntropy(pred[len(feat)-1-index-1], pred[len(feat)-1-index].detach()) * b
            
            loss = l_softmax / accumulation_steps
        
        # 그래디언트 스케일링 및 역전파
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer['softmax'])
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler.step(optimizer['softmax'])
            scaler.update()
            optimizer['softmax'].zero_grad(set_to_none=True)
        
        running_loss += loss.item() * accumulation_steps
        _, predicts = torch.max(pred[3], 1)
        correct_sum += torch.eq(predicts, targets).sum()
        
        # 메모리 최적화
        del pred, feat, pred_teacher, feat_teacher
        if i % 50 == 0:  # 주기적 캐시 정리
            torch.cuda.empty_cache()

    train_acc = correct_sum.float() / len(train_loader.dataset)
    train_loss = running_loss / iter_cnt
    
    # Validation with mixed precision
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_iter_cnt = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            pred, _, _, _ = model(images)
            loss = criterion['softmax'](pred[3], targets)
            
            val_loss += loss.item()
            _, predicts = torch.max(pred[3], 1)
            val_correct += torch.eq(predicts, targets).sum()
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicts.cpu().numpy())
            val_iter_cnt += 1
            
            del pred
    
    val_acc = val_correct.float() / len(val_loader.dataset)
    val_loss = val_loss / val_iter_cnt
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, B-Acc: {balanced_acc:.4f}')
    
    if val_loss < best_loss:
        previous_checkpoint = None
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith(f"{model_name}_{dataset_name}_iter{iteration+1}_epoch") and filename.endswith(".pth"):
                previous_checkpoint = os.path.join(checkpoint_dir, filename)
                break

        if previous_checkpoint and os.path.exists(previous_checkpoint):
            os.remove(previous_checkpoint)

        best_checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{model_name}_{dataset_name}_iter{iteration+1}_epoch{epoch}_acc{val_acc:.4f}_bacc{balanced_acc:.4f}.pth"
        )
        
        # 체크포인트에 scaler 상태 포함
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer['softmax'].state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, best_checkpoint_path)

        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    should_stop = patience_counter >= patience
    if should_stop:
        print("Early stopping triggered")

    torch.cuda.empty_cache()
    return val_loss, val_acc, balanced_acc, should_stop, best_loss, patience_counter, best_checkpoint_path

def test(test_loader, model, checkpoint_path, criterion, device):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_stream = torch.cuda.Stream()
    test_loss = 0.0
    correct = 0
    iter_cnt = 0
    y_true = []
    y_pred = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, targets in test_loader:
            with torch.cuda.stream(test_stream):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            
            pred, _, _, _ = model(images)
            loss = criterion['softmax'](pred[3], targets)
            
            test_loss += loss.item()
            _, predicts = torch.max(pred[3], 1)
            correct += torch.eq(predicts, targets).sum()
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicts.cpu().numpy())
            iter_cnt += 1
            
            del pred, loss

    test_acc = correct.float() / len(test_loader.dataset)
    test_loss = test_loss / iter_cnt
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return test_acc.item(), balanced_acc, test_loss

def run_train_test():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device_ids = list(range(len(args.gpu.split(','))))
    device = torch.device(f'cuda:{device_ids[0]}')
    
    # CUDA 최적화 설정 강화
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # GPU 메모리 관리 최적화
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.85, i)
            torch.cuda.empty_cache()
    
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
    checkpoint_dir = f'/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/GSDNet/checkpoint/{args.dataset_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 결과 저장을 위한 리스트
    all_accuracies = []
    all_balanced_accuracies = []
    all_val_losses = []
    all_test_losses = []
    results = []

    img_size = args.img_size
    train_transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(20),
            transforms.RandomCrop(img_size, padding=32)
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    label_file = os.path.join(args.data_path, 'label/label.lst')
    image_dir = os.path.join(args.data_path, 'aligned_image')
    
    labels_data = []
    file_paths = []
    
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_name = parts[0]
            label = int(parts[-1])
            
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                labels_data.append(label)
                file_paths.append(img_path)
    
    file_paths = np.array(file_paths)
    labels = np.array(labels_data)

    # Cross validation setup
    ss = ShuffleSplit(n_splits=args.iterations, test_size=args.test_size, random_state=42)
    splits = list(ss.split(file_paths, labels))

    for iteration, (train_val_indices, test_indices) in enumerate(splits):
        print(f"\n{'='*50}")
        print(f"Starting Iteration {iteration + 1}/{args.iterations}")
        print(f"{'='*50}")

        control_random_seed(iteration)

        train_indices, val_indices = train_test_split(train_val_indices, test_size=args.val_size, random_state=iteration)

        train_dataset = ExpWDataset(
            file_paths[train_indices],
            labels[train_indices],
            transform=train_transform
        )
        val_dataset = ExpWDataset(
            file_paths[val_indices],
            labels[val_indices],
            transform=val_transform
        )
        test_dataset = ExpWDataset(
            file_paths[test_indices],
            labels[test_indices],
            transform=val_transform
        )

        print(f'Train set size: {len(train_dataset)}')
        print(f'Validation set size: {len(val_dataset)}')
        print(f'Test set size: {len(test_dataset)}')

        # Create data loaders with optimizations
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(os.cpu_count() // 2, 16),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count() // 2, 16),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count() // 2, 16),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )

        # Initialize model and training components
        model = GSDNet(num_class=args.num_class)
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)
        
        criterion = { 'softmax': nn.CrossEntropyLoss().to(device) }
        optimizer = { 'softmax': torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) }
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer['softmax'], T_max=args.t_max)

        best_loss = float('inf')
        patience_counter = 0
        best_checkpoint_path = None

        for epoch in range(1, args.epochs + 1):
            val_loss, val_acc, balanced_acc, should_stop, best_loss, patience_counter, current_checkpoint_path = train(
                train_loader, val_loader, model, criterion,
                optimizer, device, epoch, args.a, args.b,
                checkpoint_dir, args.dataset_name, args.model_name, iteration,
                best_loss, patience_counter, args.patience
            )
            
            if current_checkpoint_path:
                best_checkpoint_path = current_checkpoint_path
                
            scheduler.step()

            if should_stop:
                print("Early stopping triggered")
                break

        if best_checkpoint_path:
            print(f"\n{'#'*20} TEST PHASE {'#'*20}")
            print(f"Loading best model from: {os.path.basename(best_checkpoint_path)}")
            
            test_acc, test_balanced_acc, test_loss = test(
                test_loader, model, best_checkpoint_path, criterion, device
            )
            
            all_accuracies.append(test_acc)
            all_balanced_accuracies.append(test_balanced_acc)
            all_val_losses.append(val_loss)
            all_test_losses.append(test_loss)
            
            results.append({
                'Iteration': iteration + 1,
                'Test Accuracy': test_acc,
                'Balanced Accuracy': test_balanced_acc,
                'Val Loss': val_loss,
                'Test Loss': test_loss
            })
            
            print(f"\nIteration {iteration + 1} Test Results:")
            print(f"{'='*30}")
            print(f"Test Accuracy:      {test_acc:.4f}")
            print(f"Balanced Accuracy:  {test_balanced_acc:.4f}")
            print(f"Validation Loss:    {val_loss:.4f}")
            print(f"Test Loss:          {test_loss:.4f}")
            print(f"{'='*30}\n")

    # 최종 결과 출력 및 저장
    if len(all_accuracies) > 0:
        print(f"\n{'#'*20} FINAL RESULTS {'#'*20}")
        print(f"Number of completed iterations: {len(all_accuracies)}")
        print("\nAverage Performance:")
        print(f"{'='*30}")
        print(f"Test Accuracy:      {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
        print(f"Balanced Accuracy:  {np.mean(all_balanced_accuracies):.4f} ± {np.std(all_balanced_accuracies):.4f}")
        print(f"Validation Loss:    {np.mean(all_val_losses):.4f} ± {np.std(all_val_losses):.4f}")
        print(f"Test Loss:          {np.mean(all_test_losses):.4f} ± {np.std(all_test_losses):.4f}")
        print(f"{'='*30}")
        
        # CSV 파일 저장
        results_df = pd.DataFrame(results)
        summary = pd.DataFrame([{
            'Iteration': args.iterations,
            'Test Accuracy': f"{np.mean(all_accuracies):.4f}±{np.std(all_accuracies):.4f}",
            'Balanced Accuracy': f"{np.mean(all_balanced_accuracies):.4f}±{np.std(all_balanced_accuracies):.4f}",
            'Val Loss': f"{np.mean(all_val_losses):.4f}±{np.std(all_val_losses):.4f}",
            'Test Loss': f"{np.mean(all_test_losses):.4f}±{np.std(all_test_losses):.4f}"
        }])
        
        final_results = pd.concat([results_df, summary], ignore_index=True)
        results_file = f'{args.model_name}_{args.dataset_name}_test_results.csv'
        final_results.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")

        # Save overall results
        new_result = {
            'Experiment Time': current_time,
            'Train Time': current_time,
            'Iteration': args.iterations,
            'Dataset Name': args.dataset_name.upper(),
            'Data Split': f'{int((1-args.test_size)*(1-args.val_size)*100)}:{int((1-args.test_size)*args.val_size*100)}:{int(args.test_size*100)}',
            'Model Name': args.model_name,
            'Val Loss': f"{np.mean(all_val_losses):.4f}±{np.std(all_val_losses):.4f}",
            'Test Loss': f"{np.mean(all_test_losses):.4f}±{np.std(all_test_losses):.4f}",
            'Acc': f"{np.mean(all_accuracies):.4f}±{np.std(all_accuracies):.4f}",
            'Balanced_Acc': f"{np.mean(all_balanced_accuracies):.4f}±{np.std(all_balanced_accuracies):.4f}"
        }
        
        expected_columns = ['Experiment Time', 'Train Time', 'Iteration', 'Dataset Name', 'Data Split', 
                          'Model Name', 'Val Loss', 'Test Loss', 'Acc', 'Balanced_Acc']

        results_path = '/userHome/userhome1/automl_undergraduate/FER_Models/total_results.csv'
        
        try:
            total_results_df = pd.read_csv(results_path)
            if total_results_df.empty or not all(col in total_results_df.columns for col in expected_columns):
                total_results_df = pd.DataFrame(columns=expected_columns)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            total_results_df = pd.DataFrame(columns=expected_columns)
        
        total_results_df = pd.concat([total_results_df, pd.DataFrame([new_result])], ignore_index=True)
        total_results_df.to_csv(results_path, index=False)
        print(f"\nResults have been appended to {results_path}")
    else:
        print("\nNo results were collected during the iterations.")
        

if __name__ == "__main__":
    run_train_test()