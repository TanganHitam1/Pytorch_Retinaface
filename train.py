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
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
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
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)
# print("Printing net...")
# print(net)

if args.resume_net is not None:
    lr = 1e-4
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
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

# # optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
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

def train(epoch):
    
    net.train()
    print('Loading Dataset...')
    
    epoch = int(epoch)

    dataset = WiderFaceDetection(training_dataset,preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
        text = args.resume_net.split('/')[-1]
        text = text.split('_')[-2]
        # print(text)
        lr = float(text[2:])
        # print(lr)
    else:
        start_iter = 0
    loss_values2 =[]
    loss_values1 = []
    llv1 = []
    llv2 = []
    lcv1 = []
    lcv2 = []
    llmv1 = []
    llmv2 = []
    i=0
    training = tqdm(range(start_iter, max_iter), desc=f'Epoch: {epoch+1}')
    for iteration in training:
        # print('Epoch: ', epoch+1)
        if iteration % epoch_size == 0:
            training.set_description(f'Epoch: {epoch+1}')
            logger.info(f'Epoch: {epoch+1}')
            if i != 0:
                temp_loss_values = sum(loss_values1) / len(loss_values1)
                temp_llv = sum(llv1) / len(llv1)
                temp_lcv = sum(lcv1) / len(lcv1)
                temp_llmv = sum(llmv1) / len(llmv1)
                loss_values2.append(temp_loss_values)
                llv2.append(temp_llv)
                # config_logger.empty_var_logger('llv1', llv1)
                print('localization loss: ', temp_llv)
                print('classification loss: ', temp_lcv)
                print('landmark loss: ', temp_llmv)
                print('total loss: ', temp_loss_values)
                # print('learning rate: ', lr)
                lcv2.append(temp_lcv)
                llmv2.append(temp_llmv)
                loss_values1 = []
                llv1 = []
                lcv1 = []
                llmv1 = []
                # print(loss_values2)
                np.savetxt(f'{curve_path}/loss_values_b{batch_size}_lr{lr}_opt{optimizer.__class__.__name__}.txt', loss_values2)
                np.savetxt(f'{curve_path}/localization_loss_b{batch_size}_lr{lr}_opt{optimizer.__class__.__name__}.txt', llv2)
                np.savetxt(f'{curve_path}/classification_loss_b{batch_size}_lr{lr}_opt{optimizer.__class__.__name__}.txt', lcv2)
                np.savetxt(f'{curve_path}/landmark_loss_b{batch_size}_lr{lr}_opt{optimizer.__class__.__name__}.txt', llmv2)
                
                plt.plot(loss_values2)
                plt.title('Total loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.savefig(f'{curve_path}/loss_b{batch_size}_lr{lr}_opt{optimizer.__class__.__name__}.png')
                plt.cla()
                
                plt.plot(llv2)
                plt.title('Localization loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.savefig(f'{curve_path}/localization_loss_b{batch_size}_lr{lr}_opt{optimizer.__class__.__name__}.png')
                plt.cla()
                
                plt.plot(lcv2)
                plt.title('Classification loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.savefig(f'{curve_path}/classification_loss_b{batch_size}_lr{lr}_opt{optimizer.__class__.__name__}.png')
                plt.cla()
                
                plt.plot(llmv2)
                plt.title('Landmark loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.savefig(f'{curve_path}/landmark_loss_b{batch_size}_lr{lr}_opt{optimizer.__class__.__name__}.png')
                plt.cla()
            i = 0
        i+=1
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 10 == 0 and epoch > cfg['decay1']):
                print(f'Saving model at epcoch: {epoch}')
                torch.save(net.state_dict(), weight_path + cfg['name']+ '_epoch_' + str(epoch) + '_b' + str(batch_size) + '_lr' + str(lr) + '_opt' + str(optimizer.__class__.__name__) +'.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        # print(loss_values1)
        loss_values1.append(loss.item())
        llv1.append(loss_l.item())
        lcv1.append(loss_c.item())
        llmv1.append(loss_landm.item())
        # print('total: ',loss_values1)
        # print('localization: ',llv1)
        # print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
            # .format(epoch, max_epoch, (iteration % epoch_size) + 1,
            # epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
    logger.info(llv2)
    logger.info(sum(llv1))
    logger.info(len(llv1))
    logger.info(sum(llv1) / len(llv1))
    logger.info("Training complete...")
    torch.save(net.state_dict(), weight_path + cfg['name'] + '_b' + str(batch_size) + '_lr' + str(lr) + '_opt' + optimizer.__class__.__name__ +'_Final.pth')
    return epoch
    # plt.show()
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    logger.info('Start training...')
    try:
        epoch = train(epoch)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), weight_path + cfg['name'] + '_b' + str(batch_size) + '_lr' + str(initial_lr) + '_opt' + optimizer.__class__.__name__ +'_Final.pth')
        logger.info('Training stopped by user...')
        print('Training stopped by user...')
        plt.show()
        exit()
    except Exception as e:
        logger.error(e)
        print(e)
        logger.info('Training stopped by error at epoch: ' + str(epoch))
        exit()
