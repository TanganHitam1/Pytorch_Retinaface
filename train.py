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
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--validating_dataset', default='./data/widerface/val/label.txt', help='validation dataset directory')
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
validating_dataset = args.validating_dataset
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
weight_path = f'./weights/{name}/b{batch_size}/lr{initial_lr}/opt{optimizer.__class__.__name__}/'

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

def avg(value):
    return sum(value) / len(value)

def train(epoch):
    net.train()
    print('Loading Dataset...')
    
    epoch = int(epoch)

    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))
    val_dataset = WiderFaceDetection(validating_dataset, preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
        text = args.resume_net.split('/')[-1]
        text = text.split('_')[-2]
        lr = float(text[2:])
    else:
        start_iter = 0

    loss_values_train = []
    loc_loss_train = []
    conf_loss_train = []
    landm_loss_train = []

    loss_values_val = []
    loc_loss_val = []
    conf_loss_val = []
    landm_loss_val = []

    training = tqdm(range(start_iter, max_iter), desc=f'Epoch: {epoch+1}')
    for iteration in training:
        if iteration % epoch_size == 0:
            training.set_description(f'Epoch: {epoch+1}')
            logger.info(f'Epoch: {epoch+1}')
            if iteration > 0:
                loss_values_train.append(avg(loc_loss_train))
                loc_loss_train.append(avg(loc_loss_train))
                conf_loss_train.append(avg(conf_loss_train))
                landm_loss_train.append(avg(landm_loss_train))
                
                loss_values_val.append(avg(loc_loss_val))
                loc_loss_val.append(avg(loc_loss_val))
                conf_loss_val.append(avg(conf_loss_val))
                landm_loss_val.append(avg(landm_loss_val))

                curve_plot('total loss', batch_size, lr, optimizer.__class__.__name__, loss_values_train)
                curve_plot('localization_loss', batch_size, lr, optimizer.__class__.__name__, loc_loss_train)
                curve_plot('classification_loss', batch_size, lr, optimizer.__class__.__name__, conf_loss_train)
                curve_plot('landmark_loss', batch_size, lr, optimizer.__class__.__name__, landm_loss_train)

                curve_plot('total loss_val', batch_size, lr, optimizer.__class__.__name__, loss_values_val)
                curve_plot('localization_loss_val', batch_size, lr, optimizer.__class__.__name__, loc_loss_val)
                curve_plot('classification_loss_val', batch_size, lr, optimizer.__class__.__name__, conf_loss_val)
                curve_plot('landmark_loss_val', batch_size, lr, optimizer.__class__.__name__, landm_loss_val)
                
                loc_loss_train = []
                conf_loss_train = []
                landm_loss_train = []
                
                loc_loss_val = []
                conf_loss_val = []
                landm_loss_val = []

            if (epoch % 10 == 0 and epoch > 0) or (epoch % 10 == 0 and epoch > cfg['decay1']):
                print(f'Saving model at epoch: {epoch}')
                torch.save(net.state_dict(), weight_path + cfg['name'] + '_epoch_' + str(epoch) + '_b' + str(batch_size) + '_lr' + str(lr) + '_opt' + optimizer.__class__.__name__ + '.pth')
            epoch += 1

        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # Training batch
        images, targets = next(iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)))
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        out = net(images)

        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()

        loc_loss_train.append(loss_l.item())
        conf_loss_train.append(loss_c.item())
        landm_loss_train.append(loss_landm.item())

        # Validation batch
        net.eval()
        with torch.no_grad():
            images, targets = next(iter(data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)))
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]

            out = net(images)

            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm

            loc_loss_val.append(loss_l.item())
            conf_loss_val.append(loss_c.item())
            landm_loss_val.append(loss_landm.item())
        net.train()

    logger.info(loc_loss_train)
    logger.info(sum(loc_loss_train))
    logger.info(len(loc_loss_train))
    logger.info(sum(loc_loss_train) / len(loc_loss_train))
    logger.info("Training complete...")
    torch.save(net.state_dict(), weight_path + cfg['name'] + '_b' + str(batch_size) + '_lr' + str(lr) + '_opt' + optimizer.__class__.__name__ + '_Final.pth')
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
    try:
        epoch = train(epoch)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), weight_path + cfg['name'] + '_b' + str(batch_size) + '_lr' + str(initial_lr) + '_opt' + optimizer.__class__.__name__ + '_Final.pth')
        logger.info('Training stopped by user...')
        print('Training stopped by user...')
        plt.show()
        exit()
    except Exception as e:
        logger.error(e)
        print(e)
        logger.info('Training stopped by error at epoch: ' + str(epoch))
        exit()
