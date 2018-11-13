import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets
import argparse
import time
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import shutil
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import vgg_dcf as dcfnet
import resnet_dcf as dcfresnet
import vgg
import resnet
import sys
from plotcsv import plot_csv

#model = vgg.vgg16_bn, resnet.resnet18, dcfresnet.resnet18, dcfnet.vgg16_bn
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', default="resnet.resnet18", type=str, metavar='N',
                    help='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--loss_weight', default=0.003, type=float, metavar='LW',
                    help='loss weight')
parser.add_argument('--weight_decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred).data)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




def test_net(test_loader, model, criterion, use_cuda):
    batch_time = AverageMeter()
    # softmax_losses = AverageMeter()
    losses = AverageMeter()

    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'],strict=None)
    
    test_logs=[]
    for i, (img_var, label_var) in enumerate(test_loader):
        if use_cuda:
            img_var = img_var.cuda()
            label_var = label_var.cuda()

        # compute output
        img_var, label_var = torch.autograd.Variable(img_var, volatile=True)
        output_var = F.log_softmax(model(img_var),dim=1)
        loss = criterion(output_var, label_var)

        # measure accuracy and record loss
        prec1 = accuracy(output_var.data, label_var)
        losses.update(loss.data[0], img_var.size(0))
        top1.update(prec1[0], img_var.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@ {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1))

            val_log={"epoch":epoch,"val_loss":round(losses.val,3),"val_loss_avg":round(losses.avg,3),
                    "val_prec":round(top1.val,3),"val_prec_avg":round(top1.avg, 3)}
            test_logs.append(val_log)
    testLogs = pd.DataFrame(test_logs)
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    return top1.avg,testLogs



global args, best_prec1

args = parser.parse_args()
filename = args.model + "_" + str(args.batch_size)  + "_" + str(args.lr)  + "_" + str(args.momentum)  + "_" + str(args.loss_weight)  + "_" + str(args.weight_decay)
lengths = 200
root="/app/MATH-6380p/project-2/"
model_file = root + filename + '_model_best.pth.tar'
# /media/leo/0287D1936157598A/docker_ws/docker_ws/MATH-6380p/project-2/resnet.resnet18_256_0.1_0.9_0.003_0.001_model_best.pth.tar

best_prec1 = 0
if args.model == "vgg.vgg16_bn":
    model = vgg.vgg16_bn()
if args.model == "resnet.resnet18":
    model = resnet.resnet18(pretrained=False)
if args.model == "dcfresnet.resnet18":
    model = dcfresnet.resnet18(pretrained=False)
if args.model == "dcfnet.vgg16_bn":
    model = dcfnet.vgg16_bn(pretrained=False)


#model = vgg.vgg16_bn()
# model = resnet.resnet18(pretrained=False)
# model = dcfresnet.resnet18(pretrained=False)
# model = dcfnet.vgg16_bn(pretrained=False)
model =  model.cuda()
#print(model.features.children())

val_data = datasets.CIFAR10('./CIFAR10', download=True, train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]))

num_all_val = len(val_data)

val_data,test_data = torch.utils.data.random_split(val_data, [num_all_val-lengths,lengths])

test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

nllloss = nn.NLLLoss().cuda()  # CrossEntropyLoss = log_softmax + NLLLoss
criterion = nllloss


if args.evaluate:
    test_net(test_loader, model, criterion)

testinf = []

for epoch in range(args.start_epoch, args.epochs):

    prec1,test_logs = test_net(test_loader, model, criterion, True)  # test_net
    testinf.append(test_logs)
    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)

testLogs = pd.concat(testinf)

testLogs.to_csv(root+"test_"+filename+".csv", encoding='utf-8')

plot_csv(root+"train_"+filename +".csv",root+"test_"+filename+".csv" )