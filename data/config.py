# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 8,
    'ngpu': 1,
    # 'epoch': 250,
    # 'decay1': 190,
    # 'decay2': 220,
    'epoch': 10,
    'decay1': 7,
    'decay2': 9,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 16,
    'ngpu': 1,
    'epoch': 10,
    'decay1': 7,
    'decay2': 9,
    'image_size': 840,
    'pretrain': False,
    'return_layers': {'layer2': 'layer2', 'layer3': 'layer3', 'layer4': 'layer4'},
    'in_channel': 256,
    'out_channel': 256
}

