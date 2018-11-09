import os
import torch

# from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets
import argparse
import time
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import vgg_dcf as dcfnet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--loss_weight', default=0.003, type=float, metavar='LW',
                    help='loss weight')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
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


def validate(val_loader, model, criterion,use_cuda):
    batch_time = AverageMeter()
    #softmax_losses = AverageMeter()
    losses = AverageMeter()

    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (img_var, label_var) in enumerate(val_loader):
        if i == 0:

            if use_cuda:
                img_var = img_var.cuda()
                label_var = label_var.cuda()


            # compute output
            img_var, label_var = torch.autograd.Variable(img_var,volatile=True), torch.autograd.Variable(label_var,volatile=True)
            output_var = model(img_var)
            loss = criterion(output_var, label_var)
            # loss = criterion( F.log_softmax(output,dim=1) , label)

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
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1))
        else:
            continue               

    return top1.avg
def train(model,train_loader,test_loader,criterion,optimizer,epoch):
    print("Training... Epoch = %d" % epoch)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    #softmax_losses = AverageMeter()
    #center_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    ip1_loader = []
    idx_loader = []

    model.train()
    end = time.time()

    for i, (img, label) in enumerate(train_loader):
        #img = img.cuda()
        #label = label.cuda()

        img, label = Variable(img).cuda(), Variable(label).cuda()
        #print(img.shape)
        #print(label.shape)

        output = model(img)
        #g=make_dot(output)
        #g.view()
        #exit()
        #print('size of img',img.shape)
        #print('size of label',label.shape)




        loss = criterion( F.log_softmax(output,dim=1) , label)
        #print(loss)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        idx_loader.append((label))
        prec1 = accuracy(output.data, label)
        losses.update(loss.data[0], img.size(0))
        top1.update(prec1[0], img.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@ {top1.val:.3f} ({top1.avg:.3f})\t'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1))
            prec1 = validate(test_loader, model,criterion,True) 
            model.train()
    

    #feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)

global args, best_prec1
args = parser.parse_args()
best_prec1 = 0
model = dcfnet.vgg16_bn(pretrained=False).cuda()
#print(model.features.children())


trainset = datasets.CIFAR10('./CIFAR10', download=True, train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]))
testset = datasets.CIFAR10('./CIFAR10', download=True, train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]))

train_loader = DataLoader(trainset, batch_size=256, shuffle=True, drop_last=True , num_workers=4)
test_loader = DataLoader(testset, batch_size=256, shuffle=False, drop_last=True , num_workers=4)

nllloss = nn.NLLLoss()  # CrossEntropyLoss = log_softmax + NLLLoss

criterion = nllloss

# optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005)
optimizer = optim.Adam(model.parameters(),lr = 1e-3)

#optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1, momentum=0.9, weight_decay=0.0005)
sheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.8)  # 30 0.1

for epoch in range(args.start_epoch, args.epochs):
    sheduler.step()

    train(model, train_loader,test_loader, criterion, optimizer, epoch + 1)  # train

