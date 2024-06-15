from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
from matplotlib import pyplot as plt
import numpy as np
import config_logger
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
# parser.add_argument('--validating_dataset', default='./data/widerface/test/label.txt', help='validation dataset directory')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--optimizer', default='sgd', help='Optimizer for training')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = args.batch_size
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum

weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
validating_dataset = args.training_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)

if args.resume_net is not None:
    lr = 1e-4
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True

if args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
elif args.optimizer == 'adamax':
    optimizer = optim.Adamax(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
elif args.optimizer == 'adamw':
    optimizer = optim.AdamW(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
elif args.optimizer == 'lbfgs':
    optimizer = optim.LBFGS(net.parameters(), lr=initial_lr)
elif args.optimizer == 'rprop':
    optimizer = optim.Rprop(net.parameters(), lr=initial_lr)
elif args.optimizer == 'asgd':
    optimizer = optim.ASGD(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
else:
    raise ValueError('Invalid optimizer')

criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

name = cfg['name']

logger = config_logger.logger_config(f'./log/{name}_train_b{batch_size}_lr{initial_lr}_opt{optimizer.__class__.__name__}.txt')
logger.info('Start training...')
logger.info('Batch size: ' + str(batch_size))
logger.info('Learning rate: ' + str(initial_lr))
logger.info('Optimizer: ' + optimizer.__class__.__name__)

curve_path = f'./curve/{name}/b{batch_size}/lr{initial_lr}/opt{optimizer.__class__.__name__}'
weight_path = f'./weights/{name}/b{batch_size}/lr{initial_lr}/opt{optimizer.__class__.__name__}'

if not os.path.exists(curve_path):
    os.makedirs(curve_path)
if not os.path.exists(weight_path):
    os.makedirs(weight_path)

epoch = 0 + args.resume_epoch

def curve_plot(fig_name, batch_size, lr, optimizer_name, var_name):
    np.savetxt(f'{curve_path}/{var_name}_b{batch_size}_lr{lr}_opt{optimizer_name}.txt', var_name)
    plt.plot(var_name)
    plt.title(f'{fig_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{curve_path}/{fig_name}_b{batch_size}_lr{lr}_opt{optimizer_name}.png')
    plt.cla()

def avg(xx, y):
    average = []
    for x in xx:
        average.append(x/y)
    return average

def saving_txt(var_list, batch_size, lr, optimizer_name):
    for var_name in var_list:
        np.savetxt(f'{curve_path}/{var_name}_b{batch_size}_lr{lr}_opt{optimizer_name}.txt', var_name)

def plotting(list_train, list_val, plot_names, batch_size, lr, optimizer_name):
    for i in range(len(list_train)):
        plt.plot(list_train[i], label='train')
        plt.plot(list_val[i], label='val')
        plt.title(plot_names[i])
        plt.xlabel('Epoch')
        if i == (len(list_train)-1):
            plt.ylabel('Accuracy')
        else:
            plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{curve_path}/{plot_names[i]}_b{batch_size}_lr{lr}_opt{optimizer_name}.png')
        plt.cla()

def append_list(avg_list, all_list):
    for i in range(len(avg_list)):
        all_list[i].append(avg_list[i])
    return all_list

def save_model(net:RetinaFace, epoch):
    torch.save(net.state_dict(), f"{weight_path}_{epoch}.pth")

def train(epoch):
    net.train()
    print('Loading Dataset...')
    
    epoch = int(epoch)
    
    validation_split = 0.2
    

    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = (int(np.floor(validation_split * dataset_size)))
    
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = data.SubsetRandomSampler(train_indices)
    val_sampler = data.SubsetRandomSampler(val_indices)
    
    train_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, collate_fn=detection_collate)
    val_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, collate_fn=detection_collate)

    # epoch_size = math.ceil(len(dataset) / batch_size)

    list_loss_values_train, list_loc_loss_train, list_conf_loss_train, list_landm_loss_train = [], [], [], []
    list_loss_values_val, list_loc_loss_val, list_conf_loss_val, list_landm_loss_val = [], [], [], []
    list_acc_train, list_acc_val = [], []
    list_train = [list_loss_values_train, list_loc_loss_train, list_conf_loss_train, list_landm_loss_train, list_acc_train]
    list_val = [list_loss_values_val, list_loc_loss_val, list_conf_loss_val, list_landm_loss_val, list_acc_val]

    best_acc_val = 0
    best_loss_val = 10000

    tqdm_epoch = tqdm(range(epoch, max_epoch), desc='Epoch')

    for epoch in tqdm_epoch:

        acc_train, loss_values_train, loc_loss_train, conf_loss_train, landm_loss_train = 0, 0, 0, 0, 0
        acc_val, loss_values_val, loc_loss_val, conf_loss_val, landm_loss_val = 0, 0, 0, 0, 0

        net.train()
        tqdm_train = tqdm(train_loader, desc='Training')
        for images, targets in tqdm_train:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]

            out = net(images)

            optimizer.zero_grad()
            (loss_l, loss_c, loss_landm), out = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            loss.backward()
            optimizer.step()

            loss_values_train += loss.item()
            loc_loss_train += loss_l.item()
            conf_loss_train += loss_c.item()
            landm_loss_train += loss_landm.item()

            _, predicted = torch.max(out[1], -1)
            total = targets[1].size(0)
            correct = (predicted == targets[1]).sum().item()
            acc_train += correct / total

        net.eval()
        with torch.no_grad():
            tqdm_val = tqdm(val_loader, desc='Validation')
            for images, targets in tqdm_val:
                images = images.cuda()
                targets = [anno.cuda() for anno in targets]

                out = net(images)

                (loss_l, loss_c, loss_landm), out = criterion(out, priors, targets)
                loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm

                loss_values_val += loss.item()
                loc_loss_val += loss_l.item()
                conf_loss_val += loss_c.item()
                landm_loss_val += loss_landm.item()

                _, predicted = torch.max(out[1], -1)
                total = targets[1].size(0)
                correct = (predicted == targets[1]).sum().item()
                acc_val += correct / total

        average_train = avg([loss_values_train, loc_loss_train, conf_loss_train, landm_loss_train, acc_train], len(train_loader))
        average_val = avg([loss_values_val, loc_loss_val, conf_loss_val, landm_loss_val, acc_val], len(val_loader))
        list_train = append_list(average_train, list_train)
        list_val = append_list(average_val, list_val)

        if acc_val >= best_acc_val and loss_values_val <= best_loss_val:
            best_acc_val = acc_val
            best_loss_val = loss_values_val
            save_model(net, f"best_{epoch}")
    save_model(net, "Final")
    return epoch

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** step_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    logger.info('Start training...')
    # try:
    epoch = train(epoch)
    # except KeyboardInterrupt:
    #     torch.save(net.state_dict(), weight_path + cfg['name'] + '_b' + str(batch_size) + '_lr' + str(initial_lr) + '_opt' + optimizer.__class__.__name__ + '_Final.pth')
    #     logger.info('Training stopped by user...')
    #     print('Training stopped by user...')
    #     plt.show()
    #     exit()
    # except Exception as e:
    #     logger.error(e)
    #     print(e)
    #     logger.info('Training stopped by error at epoch: ' + str(epoch))
    #     exit()
