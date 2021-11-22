from torch.utils.data import Dataset
import os.path as osp
import numpy as np
import json
import torch
import argparse
import glob
import vedacore.fileio as fileio
import vedacore.image as image
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--dataset_file', type=str, default='')
    parser.add_argument('--enc_layers', default=512, type=int)
    parser.add_argument('--frames_root', default='')
    args = parser.parse_args()
    return args

class Thumos14Dataset(Dataset):
    def __init__(self, args, flag='train'):
        assert flag in ['train', 'test', 'valid']
        # self.pickle_root = args.data_root
        self.sessions = getattr(args, flag + '_session_set')
        self.training = flag == 'train'
        self.enc_steps = args.enc_layers
        self.frames_root = args.frames_root
        self.stride = args.stride if self.training else args.stride*2
        self.inputs = []
        self.feature_All = {}
        self.frames_all = {}
        self.subnet = 'val' if self.training else 'test'
        self.dataset_file = osp.join(args.dataset_file, self.subnet + '.json')
        self.class_gt = {"Background": 0, "BaseballPitch": 1, "BasketballDunk": 2, "Billiards": 3,
                         "CleanAndJerk": 4, "CliffDiving": 5, "CricketBowling": 6, "CricketShot": 7,
                         "Diving": 8, "FrisbeeCatch": 9, "GolfSwing": 10, "HammerThrow": 11,
                         "HighJump": 12, "JavelinThrow": 13, "LongJump": 14, "PoleVault": 15,
                         "Shotput": 16, "SoccerPenalty": 17, "TennisSwing": 18, "ThrowDiscus": 19,
                         "VolleyballSpiking": 20, "Ambiguous": 21}

        for session in self.sessions:
            imgfiles = sorted(glob.glob(osp.join(self.frames_root, self.subnet, session, '*')))
            num_imgs = len(imgfiles)
            self.frames_all[session] = imgfiles
            seed = np.random.randint(64) if self.training else 0
            class_json = json.load(open(self.dataset_file, 'rb'))
            feature_num = math.floor(num_imgs/8)

            target_list = [0 for _ in range(feature_num)]

            duration = class_json['database'][session]['duration']
            for video_info in class_json['database'][session]['annotations']:
                label = video_info['label']
                segment_t = video_info['segment']
                target_list = self.annotation(label, segment_t, duration, target_list)
            target = np.eye(22)[target_list]
            for start, end in zip(
                    range(seed*8, num_imgs, 8*self.stride),
                    range(seed*8 + self.enc_steps, num_imgs, 8*self.stride)):
                enc_target = target[math.floor(start/8):math.floor(end/8)]
                class_h_target = enc_target[64 - 1]
                if class_h_target.argmax() != 21:
                    self.inputs.append([session, start, end, enc_target])
        print('1')

    def annotation(self, label, segment, duration, target):
        if segment[0] > duration or segment[1] > duration:
            return target
        begin = round((float(segment[0]) / float(duration)) * len(target))
        end = round((float(segment[1]) / float(duration)) * len(target))
        if end == len(target):
            end -= 1
        for i in range(begin, end + 1, 1):
            if target[i] == 5 and self.class_gt[label] == 8:
                continue
            else:
                target[i] = self.class_gt[label]
        return target

    def load_frame(self, frame_file):
        x = fileio.FileClient()
        img_bytes = x.get(frame_file)
        img = image.imfrombytes(img_bytes, flag='color')
        img = img.astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        data = image.imnormalize(img, mean=mean, std=std, to_rgb=True)

        return data

    def __getitem__(self, index):
        session, start, end, enc_target = self.inputs[index]

        enc_target = torch.tensor(enc_target)

        imgs = []
        for i in range(start, end):
            img = self.load_frame(self.frames_all[session][i + 1])
            imgs.append(img)
        imgs = np.array(imgs)
        imgs = torch.tensor((imgs))
        return imgs, enc_target

    def __len__(self):
        return len(self.inputs)

if __name__ == '__main__':
    args = parse_args()

    with open('/disk/sunyuan/1105/new_online /data/data_info_new.json', 'r') as f:
        data_info = json.load(f)['THUMOS']

    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']
    args.class_index = data_info['class_index']
    args.numclass = len(args.class_index)

    dataset_train = Thumos14Dataset(args, 'train')
    dataset_test = Thumos14Dataset(args, 'test')
