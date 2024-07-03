"""
the general training framework
"""

from __future__ import print_function

import os
import re
import argparse
import time

import numpy
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models_da import model_dict
from distiller_zoo.mergekd import SELayer_Single

from models_da.Merge_model_da import Merge_model_da

from helper.loops_da import train_distill_adapter as train, validate_distill, validate_vanilla
from helper.util import save_dict_to_json, reduce_tensor, TimeConverter
from distiller_zoo import DistillKL
from dataset.DA_dataset import get_train_test_loader

split_symbol = '~' if os.name == 'nt' else ':'


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=120, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='40,70,100', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset and model
    parser.add_argument('--data_type', type=str, default='officehome',
                        choices=['digit', 'office31', 'officehome', 'visda', 'pacs', 'officecaltech', 'domainnet'],
                        help='dataset')
    parser.add_argument('--data_path', type=str, default='./data/DA_dataset/officehome/Clipart')
    parser.add_argument('--dataset', type=str, default='Clipart', help='dataset')
    parser.add_argument('--t_dataset', type=str, default='Clipart', help='dataset')
    parser.add_argument('--model_t', type=str, default='resnet34')
    parser.add_argument('--model_s', type=str, default='resnet18')
    parser.add_argument('--path_t', type=str,
                        default='./save/teachers/models_DA/officehome/resnet34_vanilla_Clipart_trial_1_0.001/resnet34_best.pth',
                        help='teacher model snapshot')

    # distillation
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--distill', type=str, default='orikd_sg', choices=['kd', 'orikd_sg'])
    parser.add_argument('-c', '--cls', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-d', '--div', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.1, help='weight balance for other losses')

    # multiprocessing
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation of teacher')

    opt = parser.parse_args()

    # set different learning rates for these MobileNet/ShuffleNet models
    if opt.model_s in ['MobileNetV2', 'MobileNetV2_1_0', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_1_5']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard
    # opt.model_path = './save/students/models_' + opt.model_t + '_' + opt.model_s
    opt.model_path = './save/students/models_DA' + opt.distill + '/' + opt.data_type + '/' + opt.t_dataset

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # opt.model_t = get_teacher_name(opt.path_t)

    model_name_template = split_symbol.join(['Domain', '{}_S', '{}_T', '{}_{}_r', '{}_a', '{}_b', '{}_trial_{}'])
    opt.model_name = model_name_template.format(opt.dataset, opt.model_s, opt.model_t, opt.distill,
                                                opt.cls, opt.div, opt.beta, opt.trial)

    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    return segments[0]


def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model')
    # model_t = get_teacher_name(model_path)
    model = model_dict[opt.model_t](pretrained=False, num_classes=n_cls)
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    print('==> done')
    return model


best_acc = 0
total_time = time.time()


def main():
    opt = parse_option()

    # ASSIGN CUDA_ID
    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = 0
    opt.gpu_id = 0

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Only one node now.
        opt.rank = gpu
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if opt.deterministic:
        torch.manual_seed(12345)
        cudnn.deterministic = True
        cudnn.benchmark = False
        numpy.random.seed(12345)

    # model
    n_cls = {
        'digit': 10,
        'office31': 31,
        'officehome': 65,
        'cinic': 10,
        'cifar100': 100,
        'imagenet': 1000,
        'officecaltech': 10,
        'pacs': 7,
        'domainnet': 345,
    }.get(opt.data_type, None)

    model_temp = load_teacher(opt.path_t, n_cls, opt.gpu, opt)

    model_s = model_dict[opt.model_s](pretrained=True, num_classes=1000)
    num_features = model_s.fc.in_features
    model_s.fc = torch.nn.Linear(num_features, n_cls)



    if opt.data_type == 'officehome':
        data = torch.randn(2, 3, 224, 224)
    elif opt.data_type == 'office31':
        data = torch.randn(2, 3, 224, 224)
    else:
        data = torch.randn(2, 3, 224, 224)
    model_temp.eval()
    model_s.eval()

    if 'vgg' in opt.model_t:
        feat_s, _ = model_s.extract_feature(data, preReLU=True, model=opt.model_s)
        feat_t, _ = model_temp.extract_feature(data, preReLU=True, model=opt.model_t)
    else:
        feat_t, _ = model_temp.extract_feature(data)
        feat_s, _ = model_s.extract_feature(data)

    feat_dim = []
    for feat in feat_t:
        feat_dim.append(feat.shape[1])

    model_t = Merge_model_da(model_temp, feat_dim)


    for name, param in model_t.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False


    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    num_channel = 0
    for feat in feat_s[1:-1]:
        num_channel += feat.shape[1]
    criterion_kd = DistillKL(opt.kd_T)
    se = SELayer_Single(num_channel)
    module_list.append(se)
    trainable_list.append(se)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    module_list.append(model_t)

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                module_list.cuda(opt.gpu)
                distributed_modules = []
                for module in module_list:
                    DDP = torch.nn.parallel.DistributedDataParallel
                    distributed_modules.append(DDP(module, device_ids=[opt.gpu]))
                module_list = distributed_modules
                criterion_list.cuda(opt.gpu)
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion_list.cuda()
            module_list.cuda()
        if not opt.deterministic:
            cudnn.benchmark = True

    # dataloader
    if opt.data_type == 'office31':
        train_loader, val_loader = get_train_test_loader(opt.data_type, opt.data_path, batch_size=opt.batch_size,
                                                          num_workers=opt.num_workers)
    elif opt.data_type == 'officehome':
        train_loader, val_loader = get_train_test_loader(opt.data_type, opt.data_path, batch_size=opt.batch_size,
                                                          num_workers=opt.num_workers)
    elif opt.data_type == 'officecaltech':
        train_loader, val_loader = get_train_test_loader(opt.data_type, opt.data_path, batch_size=opt.batch_size,
                                                          num_workers=opt.num_workers)
    elif opt.data_type == 'pacs':
        train_loader, val_loader = get_train_test_loader(opt.data_type, opt.data_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.data_type == 'domainnet':
        train_loader, val_loader = get_train_test_loader(opt.data_type, opt.data_path, batch_size=opt.batch_size, num_workers=opt.num_workers)

    else:
        raise NotImplementedError(opt.dataset)

    optimizer_adapter = optim.AdamW(model_t.parameters(), lr=0.001, eps=1e-4)
    scheduler_adapter = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adapter, opt.epochs * len(train_loader))


    # routine
    for epoch in range(1, opt.epochs + 1):
        s_time = time.time()
        torch.cuda.empty_cache()

        print("==> training...")

        time1 = time.time()
        train_acc, train_acc_top5, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, optimizer_adapter, scheduler_adapter, opt)
        teacher_acc, _, _ = validate_vanilla(val_loader, model_t, criterion_cls, opt)
        time2 = time.time()

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' * Epoch {}, GPU {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}'.format(epoch, opt.gpu, train_acc,
                                                                                        train_acc_top5, time2 - time1))


        print('GPU %d validating' % (opt.gpu))
        test_acc, test_acc_top5, test_loss = validate_distill(val_loader, module_list, criterion_cls, opt)
        e_time = time.time()
        t_time = e_time - s_time
        text = TimeConverter(t_time, epoch, opt.epochs)
        print(text)
        if opt.dali is not None:
            train_loader.reset()
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_acc': best_acc,
                }
                if opt.distill == 'simkd':
                    state['proj'] = trainable_list[-1].state_dict()
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))

                test_merics = {'test_loss': test_loss,
                               'test_acc': test_acc,
                               'test_acc_top5': test_acc_top5,
                               'epoch': epoch}

                save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))
                print('saving the best model!')
                torch.save(state, save_file)

    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # This best accuracy is only for printing purpose.
        print('best accuracy:', best_acc)
        # save parameters
        save_state = {k: v for k, v in opt._get_kwargs()}
        # No. parameters(M)
        num_params = (sum(p.numel() for p in model_s.parameters()) / 1000000.0)
        save_state['Total params'] = num_params
        save_state['Total time'] = (time.time() - total_time) / 3600.0
        params_json_path = os.path.join(opt.save_folder, "parameters.json")
        save_dict_to_json(save_state, params_json_path)


if __name__ == '__main__':
    main()
