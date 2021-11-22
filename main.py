import datetime
import json
import time
import os.path as osp
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from Loss_Evaluate.evaluate import SetCriterion
import init as pre
from network.resnet3d import ResNet3d
from network.Colar_dynamic import RoTal_dynamic
from tools.train import train_one_epoch
from tools.utils import backup_code, save_on_master
from Dataset.Colar_end_to_end import Thumos14Dataset


def main(args):
    log_file = backup_code(args.exp_name)
    seed = args.seed + pre.get_rank()
    pre.set_seed(seed)

    device = torch.device('cuda:' + str(args.cuda_id))
    model_res = ResNet3d(depth=50,
                         out_indices=(3,),
                         norm_eval=True,
                         frozen_stages=args.frozen_stages,
                         inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                         zero_init_residual=False)
    model_dynamic = RoTal_dynamic(args.input_size, args.numclass)

    model_dynamic.load_state_dict(torch.load('./models/weight_init.pth')['model_dynamic'])
    model_dynamic.to(device)
    model_res.load_state_dict(torch.load('./models/weight_init.pth')['model_res'])
    model_res.to(device)

    criterion = SetCriterion().to(device)
    optimizer = torch.optim.Adam([
        {"params": model_res.parameters()},
        {"params": model_dynamic.parameters()}],
        lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = Thumos14Dataset(flag='train', args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   pin_memory=True, num_workers=args.num_workers)

    # dataset_val = Thumos14Dataset(flag='test', args=args)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
    #                              drop_last=False, pin_memory=True, num_workers=args.num_workers)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, loss = train_one_epoch(
            model_dynamic,
            model_res,
            criterion, data_loader_train, optimizer, device, args.clip_max_norm)
        lr_scheduler.step()
        print('epoch:{}------loss:{}'.format(epoch, train_loss))
        output_dir = Path(args.output_dir)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                save_on_master({
                    'model_dynamic': model_dynamic.state_dict(),
                    'model_res': model_res.state_dict()
                }, checkpoint_path)

        # test_stats = evaluate(
        #     model_dynamic,
        #     model_res,
        #     data_loader_val, device, args, dataset_val)
        # print('---------------Calculation of the map-----------------')
        # RoTAD_evaluate(test_stats, epoch, args.command, log_file)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if '__main__' == __name__:
    args = pre.parse_args()
    data_info_path = osp.join(args.dataset_file, 'data_info.json')
    with open(data_info_path, 'r') as f:
        data_info = json.load(f)['THUMOS']

    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']
    args.class_index = data_info['class_index']
    args.numclass = len(args.class_index)
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
