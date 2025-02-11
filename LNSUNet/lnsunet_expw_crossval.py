import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import numpy as np
import random
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from datetime import datetime

from backbones.swin import SwinTransformer
from expression.models import SwinTransFER

def get_args():
    parser = argparse.ArgumentParser()
    
    # GPU configuration
    parser.add_argument('--gpu', type=str, default='0,1', help='GPU id to use')
    
    # Model configuration 
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--init', action='store_true', default=True, help='Initialize model')
    parser.add_argument('--save_all_states', action='store_true', default=True, help='Save all states')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16')
    parser.add_argument('--checkpoint_dir', type=str, default='/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/LNSU-Net/checkpoints/expw', help='Checkpoint directory')
    
    # Training parameters
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience')
    
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    
    # Learning rate scheduler
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'exp'], help='Learning rate scheduler type')
    parser.add_argument('--t_max', type=int, default=100, help='T_max for CosineAnnealingLR scheduler')
    
    # Dataset configuration
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.25, help='Fraction of data to use for validation')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for repeated random sampling')
    
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

class ExpWDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path, transform=None, img_size=112, indices=None):
        self.image_paths = []
        self.labels = []
        self.data_path = data_path
        self.label_path = label_path

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([img_size, img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        # 최적화: 파일을 한 번만 읽고 리스트로 변환
        with open(self.label_path, "r") as f:
            data = [line.strip().split() for line in f.readlines()]

        all_paths = [os.path.join(self.data_path, line[0]) for line in data if os.path.exists(os.path.join(self.data_path, line[0]))]
        all_labels = [int(line[-1]) for line in data if os.path.exists(os.path.join(self.data_path, line[0]))]

        if indices is not None:
            self.image_paths = [all_paths[i] for i in indices]
            self.labels = [all_labels[i] for i in indices]
        else:
            self.image_paths = all_paths
            self.labels = all_labels

        self.labels = np.asarray(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.labels[idx])
        img1 = transforms.RandomHorizontalFlip(p=1.0)(img)
        return img, label, img1

    def __len__(self):
        return len(self.image_paths)

def ACLoss(att_map1, att_map2, grid_l, output):
    flip_grid_large = grid_l.expand(output.size(0), -1, -1, -1)
    flip_grid_large = Variable(flip_grid_large, requires_grad=False)
    flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
    att_map2_flip = F.grid_sample(att_map2, flip_grid_large, mode='bilinear', padding_mode='border', align_corners=True)
    flip_loss_l = F.mse_loss(att_map1, att_map2_flip, reduction='none')
    return flip_loss_l

def generate_flip_grid(w, h):
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float()
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid[:, 0, :, :] = -grid[:, 0, :, :]
    return grid

class LSR2(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e

    def _one_hot(self, labels, classes, value=1):
        one_hot = torch.zeros(labels.size(0), classes)
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        mask = (one_hot==0)
        balance_weight = torch.tensor([0.95124031, 4.36690391, 1.71143654, 0.25714585, 0.6191221, 1.74056738, 0.48617274]).to(one_hot.device)
        ex_weight = balance_weight.expand(one_hot.size(0),-1)
        resize_weight = ex_weight[mask].view(one_hot.size(0),-1)
        resize_weight /= resize_weight.sum(dim=1, keepdim=True)
        one_hot[mask] += (resize_weight*smooth_factor).view(-1)
        return one_hot.to(target.device)

    def forward(self, x, target):
        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)
        return torch.mean(loss)
    
def train(args, model, train_loader, val_loader, current_lr, iteration):
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    
    control_random_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model.train()
    best_val_acc = 0
    best_val_balanced_acc = 0
    patience_counter = 0
    early_stop = False

    # Optimizer 설정
    if args.optimizer == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=current_lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=current_lr,
            weight_decay=args.weight_decay
        )

    # Scheduler 설정
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.t_max)
    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    elif args.lr_scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

    # GradScaler 최적화 설정
    scaler = torch.cuda.amp.GradScaler(
        init_scale=2**16,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=args.fp16
    )

    if args.init:
        dict_checkpoint = torch.load("/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/LNSU-Net/start_0.pt")
        model.module.encoder.load_state_dict(dict_checkpoint["state_dict_backbone"], strict=True)
        del dict_checkpoint

    if args.resume:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, "last_checkpoint.pt"))
        model.load_state_dict(checkpoint["state_dict_model"])
        opt.load_state_dict(checkpoint["state_optimizer"])
        scheduler.load_state_dict(checkpoint["state_lr_scheduler"])
        scaler.load_state_dict(checkpoint["state_scaler"])  # 스케일러 상태 복원
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        patience_counter = checkpoint["patience_counter"]
    else:
        start_epoch = 0

    for epoch in tqdm(range(start_epoch, args.epochs)):
        if early_stop:
            tqdm.write(f"Early stopping triggered at epoch {epoch}")
            break

        model.train()
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0

        for idx, (expression_img, expression_label, expression_img1) in enumerate(train_loader):
            iter_cnt += 1
            if torch.cuda.is_available():
                expression_img = expression_img.cuda()
                expression_label = expression_label.cuda()
                expression_img1 = expression_img1.cuda()

            # Autocast를 사용한 forward pass
            with torch.cuda.amp.autocast(enabled=args.fp16):
                expression_output, hm = model(expression_img)
                expression_output1, hm1 = model(expression_img1)            

                grid_l = generate_flip_grid(7, 7).cuda()  
                flip_loss = ACLoss(hm, hm1, grid_l, expression_output)
                
                flip_loss = flip_loss.mean(dim=-1).mean(dim=-1)
                balance_weight = torch.tensor([0.95124031, 4.36690391, 1.71143654, 0.25714585, 0.6191221, 1.74056738, 0.48617274]).cuda().view(7,1)
                flip_loss = torch.mm(flip_loss, balance_weight).squeeze()
                
                loss = LSR2(0.3)(expression_output, expression_label) + 0.1 * flip_loss.mean()

            # Gradient scaling 및 backward pass 최적화
            scaler.scale(loss).backward()
            
            # Gradient clipping with scaler
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            
            # Optimizer step with scaler
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)  # 메모리 사용량 최적화
            
            scheduler.step()

            running_loss += loss.item()
            _, predicts = torch.max(expression_output, 1)
            correct_num = torch.eq(predicts, expression_label).sum()
            correct_sum += correct_num

            # 주기적으로 메모리 캐시 비우기
            if idx % 50 == 0:
                torch.cuda.empty_cache()

        # Calculate epoch statistics
        acc = correct_sum.float() / float(len(train_loader.dataset))
        running_loss = running_loss / iter_cnt
        tqdm.write(f'[Epoch {epoch+1}] Training accuracy: {acc:.4f}. Loss: {running_loss:.4f}')

        # Validation phase with autocast
        model.eval()
        val_loss = 0.0
        val_iter_cnt = 0
        val_correct_sum = 0
        val_sample_cnt = 0
        
        y_true = []
        y_pred = []

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16):
            for val_images, val_target, _ in val_loader:
                if torch.cuda.is_available():
                    val_images = val_images.cuda()
                    val_target = val_target.cuda()

                val_output, hm = model(val_images)
                loss = LSR2(0.3)(val_output, val_target)
                
                val_loss += loss.item()
                val_iter_cnt += 1
                _, predicts = torch.max(val_output, 1)
                correct_num = torch.eq(predicts, val_target)
                val_correct_sum += correct_num.sum().item()
                val_sample_cnt += val_output.size(0)
                
                y_true.append(val_target.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())

        val_loss = val_loss / val_iter_cnt
        val_acc = val_correct_sum / val_sample_cnt * 100
        
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        val_balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100

        current_time = datetime.now().strftime('%y%m%d_%H%M%S')
        tqdm.write(f"[{current_time}] [Epoch {epoch+1}] Validation accuracy: {val_acc:.4f}. "
                  f"bacc: {val_balanced_acc:.4f}. Loss: {val_loss:.4f}")

        # Save checkpoint with scaler state
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_balanced_acc = val_balanced_acc
            patience_counter = 0
            
            # Find and remove previous checkpoint
            previous_checkpoint = None
            for filename in os.listdir(args.checkpoint_dir):
                if filename.startswith(f"expw_iter{iteration+1}_epoch") and filename.endswith(".pth"):
                    previous_checkpoint = os.path.join(args.checkpoint_dir, filename)
                    break
            
            if previous_checkpoint and os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)
            
            # Save new best checkpoint with scaler state
            best_checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f"expw_iter{iteration+1}_epoch{epoch}_acc{val_acc:.4f}_bacc{val_balanced_acc:.4f}.pth"
            )
            
            checkpoint = {
                "epoch": epoch,
                "state_dict_model": model.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": scheduler.state_dict(),
                "state_scaler": scaler.state_dict(),  # 스케일러 상태 저장
                "best_val_acc": best_val_acc,
                "best_val_balanced_acc": best_val_balanced_acc,
                "iteration": iteration
            }
            
            torch.save(checkpoint, best_checkpoint_path)
            tqdm.write(f"New best model saved at {best_checkpoint_path}")
            tqdm.write(f"Best accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= args.early_stopping_patience:
            tqdm.write(f"Early stopping triggered after {patience_counter} epochs without improvement")
            early_stop = True

        # 에포크 끝에서 메모리 정리
        torch.cuda.empty_cache()

    return best_val_acc, epoch, best_val_balanced_acc

def test(args, model, test_loader):
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    test_loss = 0.0
    test_iter_cnt = 0
    correct_sum = 0
    sample_cnt = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for idx, (images, target, _) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()

            output, _ = model(images)
            loss = criterion(output, target)

            test_loss += loss.item()
            test_iter_cnt += 1
            _, predicts = torch.max(output, 1)
            correct_num = torch.eq(predicts, target)
            correct_sum += correct_num.sum().item()
            sample_cnt += output.size(0)

            y_true.append(target.cpu().numpy())
            y_pred.append(predicts.cpu().numpy())

            if idx % 10 == 0:
                tqdm.write(
                    f'Test: [{idx}/{len(test_loader)}]\t'
                    f'Current Loss: {loss.item():.4f}\t'
                    f'Current Acc: {(correct_sum/sample_cnt*100):.2f}%'
                )

        test_loss = test_loss / test_iter_cnt
        test_acc = correct_sum / sample_cnt * 100
        
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        test_balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100

        current_time = datetime.now().strftime('%y%m%d_%H%M%S')
        tqdm.write(f"[{current_time}] Test accuracy: {test_acc:.4f}. "
                  f"bacc: {test_balanced_acc:.4f}. Loss: {test_loss:.4f}")

        return test_acc, test_balanced_acc, test_loss

def run_train_test():
    args = get_args()
    
    # GPU configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    tqdm.write("Work on GPU: " + os.environ['CUDA_VISIBLE_DEVICES'])
    
    control_random_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Image and label paths for ExpW dataset
    image_path = "/userHome/userhome1/automl_undergraduate/FER_Models/datasets/ExpW/aligned_image"
    label_path = "/userHome/userhome1/automl_undergraduate/FER_Models/datasets/ExpW/label/label.lst"
    
    # Create a temporary dataset to get total length
    temp_dataset = ExpWDataset(image_path, label_path)
    total_indices = range(len(temp_dataset))
    
    ss = ShuffleSplit(n_splits=args.iterations, test_size=args.test_size, random_state=args.seed)
    splits = list(ss.split(total_indices))
    
    results = []

    for iteration in range(args.iterations):
        tqdm.write(f"\nIteration {iteration + 1}/{args.iterations}")
        train_val_indices, test_indices = splits[iteration]
        
        train_indices, val_indices = train_test_split(
            train_val_indices, 
            test_size=args.val_size, 
            random_state=iteration
        )

        tqdm.write(f'Train set size: {len(train_indices)}')
        tqdm.write(f'Validation set size: {len(val_indices)}')
        tqdm.write(f'Test set size: {len(test_indices)}')

        # Initialize model
        swin = SwinTransformer(num_classes=512)
        model = SwinTransFER(swin=swin, swin_num_features=768, num_classes=7, cam=True)

        # DataLoader setup
        train_loader = DataLoader(
            ExpWDataset(image_path, label_path, indices=train_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(os.cpu_count() // 2, 16),
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            ExpWDataset(image_path, label_path, indices=val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count() // 2, 16),
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            ExpWDataset(image_path, label_path, indices=test_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count() // 2, 16),
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        # Learning rate scaling
        batch_size_scale = args.batch_size / 512.0
        current_lr = args.lr * batch_size_scale

        # Train and evaluate
        best_val_acc, best_epoch, best_val_balanced_acc = train(
            args, model, train_loader, val_loader, current_lr, iteration
        )
        
        # Load best model for testing
        best_checkpoint = None
        best_checkpoint_path = None
        for filename in os.listdir(args.checkpoint_dir):
            if filename.startswith(f"expw_iter{iteration+1}_epoch") and filename.endswith(".pth"):
                if best_checkpoint_path is None or filename > best_checkpoint_path:
                    best_checkpoint_path = os.path.join(args.checkpoint_dir, filename)
        
        if best_checkpoint_path:
            best_checkpoint = torch.load(best_checkpoint_path)
            state_dict = best_checkpoint["state_dict_model"]
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        
        # Test
        test_acc, test_balanced_acc, test_loss = test(args, model, test_loader)
        
        result = {
            'Iteration': iteration + 1,
            'Test Accuracy': test_acc,
            'Balanced Accuracy': test_balanced_acc,
            'Loss': test_loss
        }
        results.append(result)

        tqdm.write(f"\nIteration {iteration + 1} Results:")
        tqdm.write(f"Test Accuracy: {test_acc:.4f}")
        tqdm.write(f"Balanced Accuracy: {test_balanced_acc:.4f}")
        tqdm.write(f"Test Loss: {test_loss:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.checkpoint_dir, "LNSUNet_ExpW_test_results.csv"), index=False)

    # Calculate summary statistics
    mean_accuracy = results_df['Test Accuracy'].mean()
    std_accuracy = results_df['Test Accuracy'].std()
    mean_balanced_accuracy = results_df['Balanced Accuracy'].mean()
    std_balanced_accuracy = results_df['Balanced Accuracy'].std()
    mean_loss = results_df['Loss'].mean()
    std_loss = results_df['Loss'].std()

    tqdm.write(f"\nFinal Results over {args.iterations} iterations:")
    tqdm.write(f"Test Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    tqdm.write(f"Balanced Accuracy: {mean_balanced_accuracy:.4f} ± {std_balanced_accuracy:.4f}")
    tqdm.write(f"Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # 최적화 적용
    torch.backends.cudnn.enabled = True  # cuDNN 가속 최적화
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 연산 활성화
    run_train_test()