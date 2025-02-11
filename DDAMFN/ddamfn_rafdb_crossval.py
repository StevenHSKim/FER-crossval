import warnings
warnings.filterwarnings('ignore')

import os
import sys
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
    parser.add_argument('--raf_path', type=str, default='/userHome/userhome1/automl_undergraduate/FER_Models/DDAMFN/datasets/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('--gpu', type=str, default='0,1', help='Assign GPUs by their numbers, e.g., "0,1" for multiple GPUs.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    
    # DDAMFN
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    
    # ------------------------------- 하이퍼 파라미터 정리 (default 값 사용) -------------------------------
    # batch size
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    
    # optimizer
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    
    # train test split ratio
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.25, help='Fraction of data to use for validation')
    
    # learning rate scheduler
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'exp'], help='Learning rate scheduler type')
    parser.add_argument('--t_max', type=int, default=20, help='T_max for CosineAnnealingLR scheduler')
    
    # iterations
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for ShuffleSplit.')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience.')
    # ------------------------------------------------------------------------------------------------
    return parser.parse_args()

# 랜덤 시드를 고정하는 함수
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

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, indices, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None, names=['name', 'label'])
        self.file_names = df['name'].values[indices]
        self.labels = df['label'].values[indices] - 1

        self.file_paths = [os.path.join(self.raf_path, 'Image/aligned', f.split(".")[0] + "_aligned.jpg") for f in self.file_names]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']  

def train(train_loader, val_loader, device, args, iteration):
    model = DDAMNet(num_class=7, num_head=args.num_head)
    model = nn.DataParallel(model)  # 모델을 DataParallel로 감싸기
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

            if acc > 0.86 and acc == best_acc:
                checkpoint_path = os.path.join('/userHome/userhome1/automl_undergraduate/FER_Models/DDAMFN/checkpoints', f"rafdb_iter{iteration+1}_epoch{epoch}_acc{acc:.4f}_bacc{balanced_acc:.4f}.pth")
                torch.save({'iter': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
                tqdm.write('Model saved.')

                best_checkpoint_path = checkpoint_path

            val_loss = running_loss

            # Early stopping check
            if best_loss >= val_loss:
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
    model = DDAMNet(num_class=7, num_head=args.num_head)
    # checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # 모델이 DataParallel로 저장되고 있으므로 아래의 전처리 필요 ('module.' 을 모두 제거)
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
    # matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    # np.set_printoptions(precision=2)
    # plt.figure(figsize=(10, 8))
    # plot_confusion_matrix(matrix, classes=class_names, normalize=True, title='RAF-DB Confusion Matrix (acc: %0.2f%%)' % (acc * 100))

    # plt.savefig(os.path.join('/userHome/userhome1/automl_undergraduate/FER_Models/DDAMFN/checkpoints', "rafdb" + "_acc" + str(acc) + "_bacc" + ".png"))
    # plt.close()

    return acc, balanced_acc, running_loss

def run_train_test():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(os.path.join(args.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None, names=['name', 'label'])
    file_names = df['name'].values
    labels = df['label'].values - 1

    ss = ShuffleSplit(n_splits=args.iterations, test_size=0.2, random_state=42)
    splits = list(ss.split(file_names, labels))

    all_accuracies = []
    best_accuracies = []
    results = []

    for iteration, (train_val_indices, test_indices) in enumerate(splits):
        print(f"Iteration {iteration + 1}/{args.iterations}")
        
        control_random_seed(iteration)

        train_indices, val_indices = train_test_split(train_val_indices, test_size=args.test_size, random_state=iteration)
        
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

        train_dataset = RafDataSet(args.raf_path, phase='train', indices=train_indices, transform=data_transforms)
        print('Whole train set size:', len(train_dataset))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)

        data_transforms_val = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])   

        val_dataset = RafDataSet(args.raf_path, phase='val', indices=val_indices, transform=data_transforms_val)
        print('Validation set size:', len(val_dataset))
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

        test_dataset = RafDataSet(args.raf_path, phase='test', indices=test_indices, transform=data_transforms_val)
        print('Test set size:', len(test_dataset))
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

        best_checkpoint_path, best_acc = train(train_loader, val_loader, device, args, iteration)
        if best_checkpoint_path:
            test_acc, test_balanced_acc, test_running_loss = test(test_loader, device, best_checkpoint_path)
            all_accuracies.append(test_acc)
            best_accuracies.append(best_acc)
            results.append([iteration + 1, test_acc, test_balanced_acc, test_running_loss])
            print(f"Test Accuracy: {test_acc}, Test Balanced Accuracy: {test_balanced_acc}, Test Running Loss: {test_running_loss}")

    results_df = pd.DataFrame(results, columns=['Iteration', 'Test Accuracy', 'Balanced Accuracy', 'Loss'])
    results_df.to_csv('DDAMFN_test_results.csv', index=False)

    print("\nBest Accuracies over all iterations:")
    for i, acc in enumerate(best_accuracies, 1):
        print(f"Iteration {i}: {acc:.4f}")

    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)

    print(f"\nMean Test Accuracy over {args.iterations} iterations: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

if __name__ == "__main__":
    run_train_test()
