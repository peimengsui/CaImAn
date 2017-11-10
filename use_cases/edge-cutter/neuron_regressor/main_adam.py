import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg

from data_loader import *

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Neuron Regressor')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11_bn',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg11_bn)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num_batchs_of_1_epoch', default=60, type=int,
                    help='num_batchs_of_1_epoch')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--valid_num_batchs_of_1_epoch', default=60, type=int,
                    help='num_batchs_of_1_epoch')


def main():
    global args
    args = parser.parse_args()

    
    train_image  = np.load('/mnt/ceph/neuro/edge_cutter/generated_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_..npz')
    train_image = train_image['arr_0']

    train_locations = pickle.load(open('/mnt/ceph/neuro/edge_cutter/generated_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_..pkl', 'rb')) 

    no_neuron_image = np.load('/mnt/ceph/neuro/edge_cutter/zero_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.test.npz')
    no_neuron_image = no_neuron_image['arr_0']

    no_neuron_locations = pickle.load(open('/mnt/ceph/neuro/edge_cutter/zero_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.test.pkl', 'rb'))


    test_image  = np.load('/mnt/ceph/neuro/edge_cutter/generated_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_._test.npz')
    test_image = test_image['arr_0']
    test_locations = pickle.load(open('/mnt/ceph/neuro/edge_cutter/generated_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_._test.pkl', 'rb')) 
    '''
    train_image = pickle.load(open(\
        '/mnt/ceph/neuro/edge_cutter/generated_input_data_sample/train_image.pkl', 'rb')) 

    train_locations = pickle.load(open(\
        '/mnt/ceph/neuro/edge_cutter/generated_input_data_sample/train_locations_sample.pkl', 'rb')) 

    test_image = pickle.load(open(\
        '/mnt/ceph/neuro/edge_cutter/generated_input_data_sample/test_image_sample.pkl', 'rb')) 

    test_locations = pickle.load(open(\
        '/mnt/ceph/neuro/edge_cutter/generated_input_data_sample/test_locations_sample.pkl', 'rb')) 

    no_neuron_image = pickle.load(open(\
        '/mnt/ceph/neuro/edge_cutter/generated_input_data_sample/no_neuron_image_sample.pkl', 'rb')) 

    no_neuron_locations = pickle.load(open(\
        '/mnt/ceph/neuro/edge_cutter/generated_input_data_sample/no_locations_sample.pkl', 'rb')) 
    '''

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = vgg.__dict__[args.arch]()
    #if torch.cuda.is_available():
    #    model.features = torch.nn.DataParallel(model.features)
    if torch.cuda.is_available():
        model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mse = checkpoint['best_mse']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, 4),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]), download=True),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer

    train_loader = data_generator(train_image, train_locations, no_neuron_image, no_neuron_locations, batch_size=128, train=True)

    val_loader = data_generator(test_image, test_locations, no_neuron_image, no_neuron_locations, batch_size=128, train=False)


    if torch.cuda.is_available():
        criterion = nn.MSELoss().cuda()
    else:
        criterion = nn.MSELoss()

    if args.half:
        model.half()
        criterion.half()

    #optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters())
    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    best_mse = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        mse = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = mse < best_mse
        best_mse = min(mse, best_mse)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mse': best_mse,
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    n_train = 0
    #for i, (input, target) in enumerate(train_loader):
    for i, (input, target) in train_loader:

        # measure data loading time
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        if torch.cuda.is_available():
            input_var = input_var.cuda()
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        #prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        #top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
       
        if n_train % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, n_train, args.batch_size*args.num_batchs_of_1_epoch, batch_time=batch_time,
                      data_time=data_time, loss=losses))

        n_train += 1
        if n_train > args.num_batchs_of_1_epoch:
            break


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    #top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    n_val = 0
    #for i, (input, target) in enumerate(val_loader):
    for i, (input, target) in val_loader:
        if torch.cuda.is_available():
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        if torch.cuda.is_available():
            input_var = input_var.cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        #prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        #top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if n_val % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      n_val, args.batch_size*args.valid_num_batchs_of_1_epoch, batch_time=batch_time, loss=losses))
        n_val += 1
        if n_val > args.valid_num_batchs_of_1_epoch:
            break

    print(' * MSE {losses.avg:.3f}'
          .format(losses=losses))

    return losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    if is_best:
        torch.save(state, filename)

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
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


if __name__ == '__main__':
    main()