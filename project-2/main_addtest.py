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


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', default="dcfresnet.resnet18", type=str, metavar='N',
                    help='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--loss_weight', default=0.003, type=float, metavar='LW',
                    help='loss weight')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
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

def save_checkpoint(state,is_best,filename):
    filename=filenames
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'model_best.pth.tar')

def train(model,train_loader,criterion,optimizer,epoch,use_cuda):
    print("Training... Epoch = %d" % epoch)

    batch_time = AverageMeter()
    #data_time = AverageMeter()
    #softmax_losses = AverageMeter()
    #center_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    #ip1_loader = []
    idx_loader = []

    model.train()
    end = time.time()
    train_logs = []
    for i, (img, label) in enumerate(train_loader):
        if use_cuda:
            img = img.cuda()
            label = label.cuda()
        #img = img.cuda()
        #label = label.cuda()

        img, label = Variable(img), Variable(label)
        #print(img.shape)
        #print(label.shape)

        output = F.log_softmax(model(img),dim=1)
        #g=make_dot(output)
        #g.view()
        #exit()
        #print('size of img',img.shape)
        #print('size of label',label.shape)
        loss = criterion(output, label)

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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@ {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

            train_log={"epoch":epoch,"train_loss":round(losses.val,3),"train_loss_avg":round(losses.avg,3),
                    "train_prec":round(top1.val,3),"train_prec_avg":round(top1.avg, 3)}
            train_logs.append(train_log)


    trainlogs = pd.DataFrame(train_logs)
    
    return trainlogs


def validate(val_loader, model, criterion, use_cuda):
    batch_time = AverageMeter()
    # softmax_losses = AverageMeter()
    losses = AverageMeter()

    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    val_logs=[]
    for i, (img_var, label_var) in enumerate(val_loader):
        if use_cuda:
            img_var = img_var.cuda()
            label_var = label_var.cuda()

        # compute output
        img_var, label_var = torch.autograd.Variable(img_var, volatile=True), torch.autograd.Variable(label_var,
                                                                                                      volatile=True)
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
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

            val_log={"epoch":epoch,"val_loss":round(losses.val,3),"val_loss_avg":round(losses.avg,3),
                    "val_prec":round(top1.val,3),"val_prec_avg":round(top1.avg, 3)}
            val_logs.append(val_log)
    valogs = pd.DataFrame(val_logs)
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    return top1.avg,valogs




args = parser.parse_args()
filename = args.model + "_" + str(args.batch_size)  + "_" + str(args.lr)  + "_" + str(args.momentum)  + "_" + str(args.loss_weight)  + "_" + str(args.weight_decay)


root="./"

if os.path.exists(root+"train_"+filename +".csv"):
    print("Already Done")
    sys.exit(0)

best_prec1 = 0
'''
if args.model == "vgg.vgg16_bn":
    model = vgg.vgg16_bn()
if args.model == "resnet.resnet18":
    model = resnet.resnet18(pretrained=False)
if args.model == "dcfresnet.resnet18":
    model = dcfresnet.resnet18()
if args.model == "dcfnet.vgg16_bn":
    model = dcfnet.vgg16_bn(pretrained=False)
'''
#exec("model = %s(pretrained=False)" % args.model)

#model = vgg.vgg16_bn()
#model = resnet.resnet18(pretrained=False)
model = dcfresnet.resnet18(pretrained=False)
#model = dcfnet.vgg16_bn(pretrained=False)
model =  model.cuda()
#print(model.features.children())

trainset = datasets.CIFAR10('./CIFAR10', download=True, train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]))
train_loader = DataLoader(trainset, batch_size=128, shuffle=True)

val_data = datasets.CIFAR10('./CIFAR10', download=True, train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]))
val_loader = DataLoader(val_data, batch_size=200, shuffle=False)
	
lengths = 300
test_data = torch.utils.data.random_split(val_data, lengths)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)


nllloss = nn.NLLLoss().cuda()  # CrossEntropyLoss = log_softmax + NLLLoss

criterion = nllloss

optimizer = optim.SGD(  filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
sheduler = lr_scheduler.StepLR(optimizer, 1, gamma=0.95)  # 30 0.1

if args.evaluate:
    validate(val_loader, model, criterion)

traininf = []
valinf = []
testinf = []

for epoch in range(args.start_epoch, args.epochs):
    sheduler.step()

    train_logs=train(model, train_loader, criterion, optimizer, epoch + 1,True)  # train
    traininf.append(train_logs)
    
    prec1,val_logs = validate(val_loader, model, criterion, True)  # validate
    valinf.append(val_logs)
    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    
    

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': 'mobile_cls',
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best,filename)
    
    

test_prec1,test_logs = validate(test_loader, model, criterion, True)
testinf.append(test_logs)

#trainf = pd.DataFrame(train_logs)
#valf = pd.DataFrame(val_logs)
trainlogs = pd.concat(traininf)
print(trainlogs)
valogs = pd.concat(valinf)
print(valogs)
testlogs = pd.concat(testinf)

trainlogs.to_csv(root+"train_"+filename +".csv", encoding='utf-8')
valogs.to_csv(root+"val_"+filename+".csv", encoding='utf-8')
testlogs.to_csv(root+"test_"+filename+".csv", encoding='utf-8')

plot_csv(root+"train_"+filename +".csv",root+"val_"+filename+".csv" )
plot_csv(root+"train_"+filename +".csv",root+"test_"+filename+".csv" )

