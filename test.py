import datetime
import json
import time
import os.path as osp
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import init as pre
from Loss_Evaluate.evaluate import RoTAD_evaluate
from network.resnet3d import ResNet3d
from network.Colar_dynamic import RoTal_dynamic
from tools.train import evaluate
from Dataset.Colar_end_to_end import Thumos14Dataset


def main(args):
    device = torch.device('cuda:' + str(args.cuda_id))
    model_res = ResNet3d(depth=50,
                         out_indices=(3,),
                         norm_eval=True,
                         frozen_stages=args.frozen_stages,
                         inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                         zero_init_residual=False)
    model_dynamic = RoTal_dynamic(args.input_size, args.numclass)

    #
    model_dynamic.load_state_dict(torch.load(args.pth)['model_dynamic'])
    model_dynamic.to(device)
    model_res.load_state_dict(torch.load(args.pth)['model_res'])
    model_res.to(device)

    dataset_val = Thumos14Dataset(flag='test', args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, pin_memory=True, num_workers=args.num_workers)

    start_time = time.time()

    log_file = 'result.txt'
    test_stats = evaluate(
        model_dynamic,
        model_res,
        data_loader_val, device)
    print('---------------Calculation of the map-----------------')
    RoTAD_evaluate(test_stats, args.command, log_file)

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
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
