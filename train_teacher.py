"""
Training a single model (student or teacher)
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from models_da import model_dict
from dataset.DA_dataset import get_train_test_loader, chekc_data, visda_train_test_loader
from helper.util import save_dict_to_json, reduce_tensor
from helper.loops_da import train_vanilla as train, validate_vanilla
from helper.util import TimeConverter


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # baisc
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2, help='num_workers')
    parser.add_argument('--epochs', type=int, default=120, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='40,70,100', help='where to decay lr, can be a list')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet34')
    parser.add_argument('--data_type', type=str, default='officehome', choices=['digit', 'office31', 'officehome', 'visda', 'pacs', 'officecaltech', 'domainnet'],
                        help='dataset')
    parser.add_argument('--data_path', type=str, default='./data/DA_dataset/officehome/Art')
    parser.add_argument('--dataset', type=str, default='Art', help='dataset')
    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)

    # multiprocessing
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                        help='url used to set up distributed training')

    opt = parser.parse_args()

    # set different learning rates for these MobileNet/ShuffleNet models
    if opt.model in ['MobileNetV2', 'MobileNetV2_1_0', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_1_5']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard
    opt.model_path = './save/teachers/models_DA/' + opt.data_type
    opt.tb_path = './save/teachers/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # set the model name
    opt.model_name = '{}_vanilla_{}_trial_{}_{}'.format(opt.model, opt.dataset, opt.trial, opt.learning_rate)
    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


best_acc = 0
total_time = time.time()


def main():
    opt = parse_option()

    # ASSIGN CUDA_ID

    ngpus_per_node = torch.cuda.device_count()
    print(ngpus_per_node)
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
        main_worker(opt.gpu_id, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    print(gpu)
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Only one node now.
        opt.rank = int(gpu)
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

    # model
    n_cls = {
        'digit': 10,
        'office31': 31,
        'officehome': 65,
        'officecaltech': 10,
        'cinic': 10,
        'cifar100': 100,
        'imagenet': 1000,
        'visda': 12,
        'pacs': 7,
        'domainnet': 345,
    }.get(opt.data_type, None)
    try:
        model = model_dict[opt.model](pretrained=True, num_classes=1000)
        if 'resnet' in opt.model:
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, n_cls)
        elif 'vgg' in opt.model:
            classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, n_cls)
            )
            model.classifier = classifier
        else:
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, n_cls)
    except KeyError:
        print("This model is not supported.")

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()

    cudnn.benchmark = True

    # dataloader
    if opt.data_type == 'office31':
        train_loader, test_loader = get_train_test_loader(opt.data_type, opt.data_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.data_type == 'officehome':
        train_loader, test_loader = get_train_test_loader(opt.data_type, opt.data_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.data_type == 'pacs':
        train_loader, test_loader = get_train_test_loader(opt.data_type, opt.data_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.data_type == 'officecaltech':
        train_loader, test_loader = get_train_test_loader(opt.data_type, opt.data_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.data_type == 'domainnet':
        train_loader, test_loader = get_train_test_loader(opt.data_type, opt.data_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.data_type == 'visda':
        train_loader, test_loader = visda_train_test_loader(opt.data_type, opt.data_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)


    chekc_data(train_loader, test_loader)
    # routine
    for epoch in range(1, opt.epochs + 1):
        s_time = time.time()
        print("==> training...")

        time1 = time.time()
        train_acc, train_acc_top5, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' * Epoch {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}'.format(epoch, train_acc, train_acc_top5,
                                                                                time2 - time1))

        test_acc, test_acc_top5, test_loss = validate_vanilla(test_loader, model, criterion, opt)

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'model': model.module.state_dict() if opt.multiprocessing_distributed else model.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))

                test_merics = {'test_loss': float('%.2f' % test_loss),
                               'test_acc': float('%.2f' % test_acc),
                               'test_acc_top5': float('%.2f' % test_acc_top5),
                               'epoch': epoch}

                save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))

                print('saving the best model!')
                torch.save(state, save_file)
        e_time = time.time()
        t_time = e_time - s_time
        text = TimeConverter(t_time, epoch, opt.epochs)
        print(text)

    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # This best accuracy is only for printing purpose.
        print('best accuracy:', best_acc)

        # save parameters
        state = {k: v for k, v in opt._get_kwargs()}

        # No. parameters(M)
        num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
        state['Total params'] = num_params
        state['Total time'] = float('%.2f' % ((time.time() - total_time) / 3600.0))
        params_json_path = os.path.join(opt.save_folder, "parameters.json")
        save_dict_to_json(state, params_json_path)


if __name__ == '__main__':
    main()
