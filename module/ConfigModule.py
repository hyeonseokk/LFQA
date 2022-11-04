'''
Config 정의해주는 곳
'''

import yaml
import json
import argparse
from os.path import join as pjoin
from types import SimpleNamespace
import logging
from pprint import pprint

import pytorch_lightning as pl

DATAPATH = '/home/mnt/hyeon/1.Dataset/LFQA/split'

class ManualArgs:
    @staticmethod
    def add_file_loading(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--save_filename', type=str, default='/mnt/md0/hyeon/checkpoints/LFQA/', help='filename')
        parser.add_argument('--train_file', type=str, default=pjoin(DATAPATH, 'processed', 'train_preprocessed_272634_top200_psg-top10_train.json'))
        parser.add_argument('--valid_file', type=str, default=pjoin(DATAPATH, 'processed', 'train_preprocessed_272634_top200_psg-top10_dev.json'))
        parser.add_argument('--cache_dir', type=str, default=pjoin(DATAPATH, 'processed', 'cache'), help='캐시파일 저장될 폴더 경로')
        parser.add_argument('--tbpath', type=str, default=None, help='tensorboard path')
        return parser

    @staticmethod
    def add_setting(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--model_type', type=str, default='bart',
                            choices=['pegasus', 'reformer', 'longformer', 'bart'])
        parser.add_argument('--ckpt_freq', type=int, nargs='+', default=[500, 2000, 5000],
                            help='어떤 step에서 저장할지')
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--max_len', type=int, default=1024, help='max seq len')
        parser.add_argument('--beam_size', type=int, default=1, help='beam_size')

        # 훈련 세팅 관련
        parser.add_argument('--gpus', type=int, default=None, help='gpu 개수')
        parser.add_argument('--num_node', type=int, default=1, help='node 개수')
        parser.add_argument('--accelerator', type=str, default=None, choices=["gpu", "ddp"], help='gpu auto tpu ipu 등등')
        parser.add_argument('--strategy', type=str, default=None, choices=["ddp", "ddp_spawn", "deepspeed"],
                            help='https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html')
        parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='fp setting')
        parser.add_argument('--plugins', type=str, default=None, choices=[None, 'deepspeed_stage_2', 'deepspeed_stage_2_offload', 
                                                                          'deepspeed_stage_3', 'deepspeed_stage_3_offload'], help='plugin setting : default=None')
        parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'FusedAdam', 'DeepSpeedCPUAdam'])
        parser.add_argument('--use_cache', action='store_true', default=False, help='true면 캐싱된 데이터 사용')

        # 훈련 하이퍼파라미터들
        parser.add_argument('--seed', type=int, default=None, help='seed')
        parser.add_argument('--ckpt_save_num', type=int, default=1, help='ckpt_save_num')
        parser.add_argument('--num_workers', type=int, default=0, help='num of worker for dataloader')
        parser.add_argument('--prev_model', type=str, default=None)
        parser.add_argument('--max_steps', type=int, default=None, help='모델 max training step 정의')
        parser.add_argument('--max_epochs', type=int, default=50, help='모델 max training epochs 정의')

        parser.add_argument('--proctitle', type=str, default=None)

        parser.add_argument('--config_manual', type=str, default='config/config_manual.yaml')
        parser.add_argument('--config_trainer', type=str, default='config/config_trainer.yaml')

        return parser

    @staticmethod
    def add_test(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--ckptpath', type=str, default='', help='테스트할 모델 체크포인트')
        parser.add_argument('--filepath', type=str, default='', help='테스트에 활용할 데이터셋')
        return parser

    @staticmethod
    def add_training(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=3e-5)
        parser.add_argument('--warmup_ratio', type=float, default=0.1)
        parser.add_argument('--accumulate_grad', type=int, default=1, help='gradient accumulation')
        return parser

    def make_manual_config(self, save='config/config_manual.yaml', parser=None):
        if parser is not None:
            parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        else:
            parser = argparse.ArgumentParser(description='manual args')
        parser = self.add_file_loading(parser)
        parser = self.add_setting(parser)
        parser = self.add_test(parser)
        parser = self.add_training(parser)

        if save is not None:  # save: 저장할 yaml파일이름
            with open(save, 'w', encoding='utf-8') as f:
                yaml.dump(parser.parse_args().__dict__, f)
        return parser


def make_trainer_args(save='config/config_trainer.yaml'):
    # os.makedirs('config', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    with open(save, 'w', encoding='utf-8') as f:
        yaml.dump(args.__dict__, f)


def object_hook(d):
    return SimpleNamespace(**d)


def _construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return SimpleNamespace(**dict(loader.construct_pairs(node)))


class Loader(yaml.Loader):
    pass


Loader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_yaml(filename):
    import os
    with open(filename, 'r', encoding='utf-8') as f:
        # tmp = yaml.load(f, Loader=yaml.FullLoader)
        tmp = yaml.load(f, Loader=Loader)
    return tmp


def pick_newargs(parser):
    args_defaults = parser.__dict__['_actions']
    args_new = parser.parse_args()
    newargs = SimpleNamespace()

    for item in args_defaults:
        if item.option_strings[-1] != '--help':
            key = item.option_strings[0]  # 어차피 한개지만.. 두개 이상이어도 이렇게 하는게 문제 x
            key = key.replace('-', ' ').strip().replace(' ', '-')
            item_default = item.default
            item_new = getattr(args_new, key)
            if item_default != item_new:
                setattr(newargs, key, item_new)

    return newargs


def merge_args(pre_args: SimpleNamespace, post_args: SimpleNamespace):
    '''
    pre_args에 post_args를 덧입히기
    '''
    for key in post_args.__dict__.keys():
        setattr(pre_args, key, getattr(post_args, key))

    return pre_args


def config_loading(parser, config_manual, config_trainer):
    '''
    1. training args로 설정했던 것에서 args 불러오고
    2. manual args로 설정한 것에서 args 붙여넣고
    3. 추가로 args설정하는 것들에 대해서 추가해줘야함
    즉 우선순위는 config_trainer < config_manual < args가 돼야함
    이렇게 하기 위해서,
    1. 훈련단계에서 직접 argument로 준 인자를 선별
    2. config_trainer, config_manual에 있는게 ㄹㅇ default로 설정됨 (단, 우선순위는 config_trainer < config_manual)
    3. 그리고 거기에다가 1에서 선별한 argument를 추가해주는 방식
    '''
    import logging
    out_args = load_yaml(config_trainer)
    out_args = merge_args(pre_args=out_args, post_args=load_yaml(config_manual))
    out_args = merge_args(pre_args=out_args, post_args=pick_newargs(parser))
    return out_args


def args_to_config(config, args):
    # args에 있는 내용들 중, config에 반영될 내용들을 넣어주기
    for key in args.__dict__:
        try:
            getattr(config, key)
        except:
            print(':)')
    return 0


def print_args(args:dict, print_ft='logging'):
    # args는 dict형태
    ft = {'logging': logging.info,
          'print': print,
          'pprint': pprint}
    if isinstance(print_ft, str):
        print_ft = ft[print_ft]

    for key in args:
        print_ft(str(key) + ': ' + str(args[key]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_manual', type=str, default='../config/config_manual.yaml')
    parser.add_argument('--config_trainer', type=str, default='../config/config_trainer.yaml')
    args = parser.parse_args()
    make_trainer_args(save=args.config_trainer)
    MA = ManualArgs().make_manual_config(save=args.config_manual)
