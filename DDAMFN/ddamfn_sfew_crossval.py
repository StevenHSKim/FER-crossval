import warnings
warnings.filterwarnings('ignore')

import os
import sys

# Add the parent directory to Python path for networks module import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import torch.nn.functional as F
from tqdm import tqdm

from networks.DDAM import DDAMNet

eps = sys.float_info.epsilon

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfew_path', type=str, default='/userHome/userhome1/automl_undergraduate/FER_Models/datasets/SFEW2.0', help='SFEW2.0 dataset path.')
    parser.add_argument('--gpu', type=str, default='0,1', help='Assign GPUs by their numbers, e.g., "0,1" for multiple GPUs.')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers.')
    
    # DDAMFN
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.25, help='Fraction of data to use for validation')
    
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'exp'], help='Learning rate scheduler type')
    parser.add_argument('--t_max', type=int, default=100, help='T_max for CosineAnnealingLR scheduler')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for ShuffleSplit.')
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

class AttentionLoss(nn.Module):
    def __init__(self, ):
        super(AttentionLoss, self).__init__()
    
    def forward(self, x):
        num_head = len(x)
        loss = 0
        cnt = 0
        if num_head > 1:
            for i in range(num_head-1):
                for j in range(i+1, num_head):
                    mse = F.mse_loss(x[i], x[j])
                    cnt = cnt+1
                    loss = loss+mse
            loss = cnt/(loss + eps)
        else:
            loss = 0
        return loss     
                
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*100, fmt)+'%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()

class SFEW2Dataset(data.Dataset):
    def __init__(self, sfew_path, phase, indices, transform=None):
        self.phase = phase
        self.transform = transform
        self.sfew_path = sfew_path

        df = pd.read_csv(os.path.join(self.sfew_path, 'sfew_2.0_labels.csv'))
        self.file_names = df['image_name'].values[indices]
        self.labels = df['label'].values[indices]

        self.file_paths = [os.path.join(self.sfew_path, 'sfew2.0_images', f) for f in self.file_names]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def train(train_loader, val_loader, device, args, iteration):
   model = DDAMNet(num_class=7, num_head=args.num_head)  # 7 classes for SFEW2.0
   model = nn.DataParallel(model)
   model.to(device)

   parameters = filter(lambda p: p.requires_grad, model.parameters())
   parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
   print('Total Parameters: %.3fM' % parameters)

   criterion_cls = torch.nn.CrossEntropyLoss()
   criterion_at = AttentionLoss()

   if args.optimizer == 'adam':
       optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   elif args.optimizer == 'adamw':
       optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   elif args.optimizer == 'sgd':
       optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

   if args.lr_scheduler == 'cosine':
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
   elif args.lr_scheduler == 'step':
       scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
   elif args.lr_scheduler == 'exp':
       scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

   best_acc = 0
   best_checkpoint_path = None
   best_loss = float('inf')
   patience_counter = 0
   patience = args.early_stopping_patience

   for epoch in tqdm(range(1, args.epochs + 1)):
       running_loss = 0.0
       correct_sum = 0
       iter_cnt = 0
       model.train()

       for imgs, targets in train_loader:
           iter_cnt += 1
           optimizer.zero_grad()

           imgs = imgs.to(device)
           targets = targets.to(device)
           
           out, feat, heads = model(imgs)
           
           loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads)  

           loss.backward()
           optimizer.step()
           
           running_loss += loss.item()
           _, predicts = torch.max(out, 1)
           correct_num = torch.eq(predicts, targets).sum()
           correct_sum += correct_num

       acc = correct_sum.float() / float(len(train_loader.dataset))
       running_loss = running_loss / iter_cnt
       tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss, optimizer.param_groups[0]['lr']))
       
       with torch.no_grad():
           running_loss = 0.0
           iter_cnt = 0
           bingo_cnt = 0
           sample_cnt = 0
           
           y_true = []
           y_pred = []

           model.eval()
           for imgs, targets in val_loader:
               imgs = imgs.to(device)
               targets = targets.to(device)
               
               out, feat, heads = model(imgs)
               loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads) 

               running_loss += loss.item()

               _, predicts = torch.max(out, 1)
               correct_num  = torch.eq(predicts, targets)
               bingo_cnt += correct_num.sum().cpu()
               sample_cnt += out.size(0)
               
               y_true.append(targets.cpu().numpy())
               y_pred.append(predicts.cpu().numpy())

               if iter_cnt == 0:
                   all_predicted = predicts
                   all_targets = targets
               else:
                   all_predicted = torch.cat((all_predicted, predicts), 0)
                   all_targets = torch.cat((all_targets, targets), 0)                  
               iter_cnt += 1        

           running_loss = running_loss / iter_cnt   
           scheduler.step()

           acc = bingo_cnt.float() / float(sample_cnt)
           acc = np.around(acc.numpy(), 4)
           best_acc = max(acc, best_acc)

           y_true = np.concatenate(y_true)
           y_pred = np.concatenate(y_pred)
           balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

           tqdm.write("[Epoch %d] Validation accuracy: %.4f. bacc: %.4f. Loss: %.3f" % (epoch, acc, balanced_acc, running_loss))
           tqdm.write("best_acc: " + str(best_acc))

           val_loss = running_loss

           if val_loss < best_loss:
               best_acc = acc
               checkpoint_dir = '/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/DDAMFN/checkpoints/sfew2.0'
               os.makedirs(checkpoint_dir, exist_ok=True)
               
               # Remove previous checkpoint for this iteration
               previous_checkpoint = None
               for filename in os.listdir(checkpoint_dir):
                   if filename.startswith(f"sfew2.0_iter{iteration+1}_epoch") and filename.endswith(".pth"):
                       previous_checkpoint = os.path.join(checkpoint_dir, filename)
                       break
                       
               if previous_checkpoint and os.path.exists(previous_checkpoint):
                   os.remove(previous_checkpoint)
                   
               # Save new checkpoint
               best_checkpoint_path = os.path.join(
                   checkpoint_dir,
                   f"sfew2.0_iter{iteration+1}_epoch{epoch}_acc{acc:.4f}.pth"
               )
               
               torch.save({
                   'epoch': epoch,
                   'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
               }, best_checkpoint_path)
               
               print('New best model saved, previous checkpoint removed.')
               best_loss = val_loss
               patience_counter = 0
           else:
               patience_counter += 1

           if patience_counter >= patience:
               print("Early stopping triggered")
               break
           
   return best_checkpoint_path, best_acc

def test(test_loader, device, model_path):
    args = parse_args()
    model = DDAMNet(num_class=7, num_head=args.num_head)  # 7 classes for SFEW2.0
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()   

    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_at = AttentionLoss()

    iter_cnt = 0
    bingo_cnt = 0
    sample_cnt = 0
    running_loss = 0.0
    y_true = []
    y_pred = []

    for imgs, targets in test_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        out, feat, heads = model(imgs)

        loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads)
        running_loss += loss.item()

        _, predicts = torch.max(out, 1)
        correct_num = torch.eq(predicts, targets)
        bingo_cnt += correct_num.sum().cpu()
        sample_cnt += out.size(0)

        y_true.append(targets.cpu().numpy())
        y_pred.append(predicts.cpu().numpy())

        if iter_cnt == 0:
            all_predicted = predicts
            all_targets = targets
        else:
            all_predicted = torch.cat((all_predicted, predicts), 0)
            all_targets = torch.cat((all_targets, targets), 0)
        iter_cnt += 1

    running_loss = running_loss / iter_cnt
    acc = bingo_cnt.float() / float(sample_cnt)
    acc = np.around(acc.numpy(), 4)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

    print(f"Test accuracy: {acc:.4f}, Balanced Accuracy: {balanced_acc:.4f}, Test Running Loss: {running_loss:.4f} ")

    return acc, balanced_acc, running_loss

def run_train_test():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(os.path.join(args.sfew_path, 'sfew_2.0_labels.csv'))
    file_names = df['image_name'].values
    labels = df['label'].values

    ss = ShuffleSplit(n_splits=args.iterations, test_size=0.2, random_state=42)
    splits = list(ss.split(file_names, labels))

    all_accuracies = []
    best_accuracies = []
    results = []

    for iteration, (train_val_indices, test_indices) in enumerate(splits):
        print(f"Iteration {iteration + 1}/{args.iterations}")
        
        control_random_seed(iteration)

        train_indices, val_indices = train_test_split(train_val_indices, test_size=args.val_size, random_state=iteration)
        
        data_transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                    transforms.RandomRotation(5),
                    transforms.RandomCrop(112, padding=8)
                ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02,0.25)),
        ])   

        train_dataset = SFEW2Dataset(args.sfew_path, phase='train', indices=train_indices, transform=data_transforms)
        print('Whole train set size:', len(train_dataset))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)

        data_transforms_val = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])   

        val_dataset = SFEW2Dataset(args.sfew_path, phase='val', indices=val_indices, transform=data_transforms_val)
        print('Validation set size:', len(val_dataset))
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

        test_dataset = SFEW2Dataset(args.sfew_path, phase='test', indices=test_indices, transform=data_transforms_val)
        print('Test set size:', len(test_dataset))
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

        best_checkpoint_path, best_acc = train(train_loader, val_loader, device, args, iteration)
        if best_checkpoint_path:
            test_acc, test_balanced_acc, test_running_loss = test(test_loader, device, best_checkpoint_path)
            all_accuracies.append(test_acc)
            best_accuracies.append(best_acc)
            results.append([iteration + 1, test_acc, test_balanced_acc, test_running_loss])
            print(f"Test Accuracy: {test_acc}, Test Balanced Accuracy: {test_balanced_acc}, Test Running Loss: {test_running_loss}")

    # Save detailed results
    results_df = pd.DataFrame(results, columns=['Iteration', 'Test Accuracy', 'Balanced Accuracy', 'Loss'])
    results_df.to_csv('DDAMFN_SFEW_test_results.csv', index=False)

    print("\nBest Accuracies over all iterations:")
    for i, acc in enumerate(best_accuracies, 1):
        print(f"Iteration {i}: {acc:.4f}")

    # Calculate means and standard deviations
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_test_loss = np.mean([result[3] for result in results])
    std_test_loss = np.std([result[3] for result in results])
    mean_balanced_acc = np.mean([result[2] for result in results])
    std_balanced_acc = np.std([result[2] for result in results])

    print(f"\nResults over {args.iterations} iterations:")
    print(f"Test Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Balanced Accuracy: {mean_balanced_acc:.4f} ± {std_balanced_acc:.4f}")
    print(f"Test Loss: {mean_test_loss:.4f} ± {std_test_loss:.4f}")

    # Get current time for experiment logging
    from datetime import datetime
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')

    # Calculate total parameters
    model = DDAMNet(num_class=7, num_head=args.num_head)
    parameters = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]) / 1_000_000

    # Prepare the new result row
    new_result = {
        'Experiment Time': current_time,
        'Train Time': current_time,
        'Iteration': args.iterations,
        'Dataset Name': 'SFEW2.0',
        'Data Split': f'60:20:20',
        'Model Name': 'DDAMFN',
        'Total Parameters': f"{parameters:.3f}M",
        'Val Loss': f"{mean_test_loss:.4f}±{std_test_loss:.4f}",
        'Test Loss': f"{mean_test_loss:.4f}±{std_test_loss:.4f}",
        'Acc': f"{mean_accuracy:.4f}±{std_accuracy:.4f}"
    }

    # Define expected columns
    expected_columns = ['Experiment Time', 'Train Time', 'Iteration', 'Dataset Name', 'Data Split', 
                        'Model Name', 'Total Parameters', 'Val Loss', 'Test Loss', 'Acc']

    # Read existing CSV file or create new one if it doesn't exist
    results_path = '/userHome/userhome1/automl_undergraduate/FER_Models/total_results.csv'
    try:
        total_results_df = pd.read_csv(results_path)
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