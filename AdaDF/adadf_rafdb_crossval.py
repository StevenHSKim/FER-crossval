import argparse
import math
import time
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import ShuffleSplit, train_test_split
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score


from model.model import create_model
from utils import Logger, AverageMeter, generate_adaptive_LD, generate_average_weights, get_accuracy, save_checkpoint

# Argument parser to parse command line arguments
parser = argparse.ArgumentParser(description='PyTorch Training')
# Training configurations
parser.add_argument('--epochs', default=100, type=int)          # Unified hyperparameters
parser.add_argument('--batch_size', default=128, type=int)      # Unified hyperparameters
parser.add_argument('--lr', default=0.001, type=float)          # Unified hyperparameters
parser.add_argument('--optimizer', default='AdamW', type=str)   # Unified hyperparameters
parser.add_argument('--weight_decay', default=1e-4, type=float) # Unified hyperparameters
parser.add_argument('--num_classes', default=7, type=int)
parser.add_argument('--num_samples', default=30000, type=int)
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
parser.add_argument('--dataset', default='raf', type=str)
parser.add_argument('--data_path', default='/userHome/userhome1/automl_undergraduate/FER_Models/datasets/raf-basic', type=str)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--iterations', default=10, type=int)
parser.add_argument('--test_size', default=0.2, type=float)
# Added early stopping argument
parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience.')

args = parser.parse_args()

best_acc = 0
best_epoch = 0

# Set device (GPU or CPU)
device = None

# Function to control random seed
def control_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set random seed for reproducibility
if args.seed is not None:
    control_random_seed(args.seed)

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

    # Initialize lists to store outputs
    outputs_list = []
    targets_list = []
    weights_list = []

    for i, (images, labels, idxs) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        idxs = idxs.to(device)

        if args.dataset == 'sfew':
            batch_size, ncrops, c, h, w = images.shape
            images = images.view(-1, c, h, w)
            labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)

        outputs_1, outputs_2, attention_weights = model(images)

        # Rank Regularization
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

        # Attention weight normalization and target distribution
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

        # Detach tensors before accumulating
        outputs_list.append(outputs_1.detach())
        targets_list.append(labels.detach())
        weights_list.append(attention_weights.detach())

        # Record metrics
        top1 = get_accuracy(outputs_1, labels, topk=(1,))[0].item()
        losses.update(loss.item(), images.size(0))
        losses_ce.update(loss_ce.item(), images.size(0))
        losses_kld.update(loss_kld.item(), images.size(0))
        losses_rr.update(RR_loss.item(), images.size(0))
        accs.update(top1, images.size(0))

    # Concatenate all accumulated tensors after the loop
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

            if args.dataset == 'sfew':
                batch_size, ncrops, c, h, w = images.shape
                images = images.view(-1, c, h, w)

            if phase == 'train':
                if args.dataset == 'sfew':
                    labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)
                outputs, _, _ = model(images)
            else:
                _, outputs, _ = model(images)
                if args.dataset == 'sfew':
                    outputs = outputs.view(batch_size, ncrops, -1)
                    outputs = torch.sum(outputs, dim=1) / ncrops

            loss = criterion(outputs, labels).mean()

            # Accuracy calculation and update
            top1 = get_accuracy(outputs, labels, topk=(1,))[0].item()
            losses.update(loss.item(), images.size(0))
            accs.update(top1, images.size(0))

            # Store predictions and targets for balanced accuracy
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100  # Convert to percentage

    return losses.avg, accs.avg, balanced_acc


def main():
    global device
   
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
   
    # logger = Logger('./results/log-' + time.strftime('%b%d_%H-%M-%S') + '.txt')
    logger.info(args)
    # writer = SummaryWriter()

    # Dataset loading 
    df = pd.read_csv(os.path.join(args.data_path, 'EmoLabel/list_patition_label.txt'),
                     sep=' ', header=None, names=['name', 'label'])
    file_names = df['name'].values
    labels = df['label'].values - 1

    ss = ShuffleSplit(n_splits=args.iterations, test_size=args.test_size, random_state=42)
    splits = list(ss.split(file_names, labels))

    # Training Phase
    for iteration, (train_val_indices, test_indices) in enumerate(splits):
        print(f"Iteration {iteration + 1}/{args.iterations}")
        # logger.info(f"Iteration {iteration + 1}/{args.iterations}")

        # Initialize per iteration
        best_acc = 0
        best_epoch = 0
        best_loss = float('inf')
        patience_counter = 0
        patience = args.early_stopping_patience

        control_random_seed(iteration)

        # Split data
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=args.test_size, random_state=iteration)

        train_loader, val_loader, test_loader = get_dataloaders(
            args.dataset, args.data_path, args.batch_size, args.num_workers,
            args.num_samples, train_indices, val_indices, test_indices)

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
        
        # logger.info('Start training.')

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
                'test_indices': test_indices
            }, is_best, iteration + 1)

            scheduler.step()

    # Testing Phase with Best Checkpoints
    print("\nStarting final test phase using best checkpoints...")
    test_results = []

    for iteration in range(1, args.iterations + 1):
        checkpoint_path = os.path.join('/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/Ada-DF/checkpoints', 
                                    f'model_best_iter{iteration}.pth')
        checkpoint = torch.load(checkpoint_path)
        
        model = create_model(args.num_classes, args.drop_rate).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        test_indices = checkpoint['test_indices']
        
        _, _, test_loader = get_dataloaders(
            args.dataset, args.data_path, args.batch_size, args.num_workers,
            args.num_samples, None, None, test_indices)
            
        test_loss, test_acc, test_balanced_acc = validate(test_loader, model, criterion, epoch='test', phase='test')
        test_results.append([iteration, test_acc, test_balanced_acc, test_loss])
        
        print(f"Iteration {iteration} - Test Acc: {test_acc:.4f}, Balanced Test Acc: {test_balanced_acc:.4f}")
        logger.info(f"Iteration {iteration} - Test Acc: {test_acc:.4f}, Balanced Test Acc: {test_balanced_acc:.4f}")

    # Final Results
    test_results = np.array(test_results)
    mean_acc = test_results[:, 1].mean()
    std_acc = test_results[:, 1].std()
    mean_balanced_acc = test_results[:, 2].mean()
    std_balanced_acc = test_results[:, 2].std()

    logger.info(f"Final Mean Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"Final Mean Balanced Test Accuracy: {mean_balanced_acc:.4f} ± {std_balanced_acc:.4f}")

    print(f"\nFinal Test Results:")
    print(f"Mean Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Mean Balanced Test Accuracy: {mean_balanced_acc:.4f} ± {std_balanced_acc:.4f}")

    results_df = pd.DataFrame(test_results, columns=['Iteration', 'Test Accuracy', 'Balanced Accuracy', 'Loss'])
    results_df.to_csv('Ada-df_final_test_results.csv', index=False)

if __name__ == '__main__':
    main()