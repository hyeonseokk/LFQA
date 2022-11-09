'''
실제 활용 단계에서 활용하게 될 inference module을 정의함
'''

import argparse
import logging
import os
from typing import Optional, Union

import numpy as np
import torch

from utils.training_utils import get_logger
from utils.file_utils import TorchFileModule


# logger.setLevel(logging.INFO)
torch.set_num_threads(1)


def find_ckpt(ckpt_path):
    # ckpt경로에서부터 best 체크포인트를 찾아 적용하는 것
    ckpt = None
    if ckpt_path is not None:
        if ckpt_path.endswith('.pt'):
            ckpt = ckpt_path
        else:  # 입력으로 폴더 경로가 주어지는 경우
            ckpt = None
            ckpt_list = os.listdir(ckpt_path)
            ckpt_list.sort(reverse=True)
            for item in ckpt_list:
                if 'best' in item:
                    if item.endswith('.pt'):
                        ckpt = item
                        break
            if (ckpt is None) and (os.listdir(ckpt_path) != []):
                ckpt = os.listdir(ckpt_path)[0]

    if ckpt is not None:
        ckpt = torch.load(os.path.join(ckpt_path, ckpt))
        return ckpt
    else:
        print('\n\n No checkpoint has been loaded! \n\n')
        logging.info('\n\n No checkpoint has been loaded! \n\n')
        return None


class GenerationModule:
    def __init__(self,
                 model,
                 tokenizer,
                 ckpt,
                 inference_args,
                 device):

        self.model = model.to(device)
        self.tokenizer = tokenizer
        if ckpt is not None:
            logging.info(self.model.load_state_dict(ckpt['model_state_dict']))
        self.model_args = ckpt['args']

        self.inference_args = inference_args
        self.device = device

        self.fileutils = TorchFileModule()
        self._set_variables()

    def _set_variables(self):
        # 필요한 변수 선언
        self.max_len = self.inference_args.max_len
        self.beam = self.inference_args.beam
        self.top_p = self.inference_args.top_p
        self.top_k = self.inference_args.top_k
        self.temperature = self.inference_args.temperature
        self.length_penalty = self.inference_args.length_penalty
        self.repetition_penalty = self.inference_args.repetition_penalty
        self.no_repeat_ngram_size = self.inference_args.no_repeat_ngram_size
        self.num_generate = self.inference_args.num_generate
        self.enc_dec = True  # 다른 모델 쓸거면 조건 추가해서 달아줄것

    @torch.no_grad()
    def generate(self, instance):
        '''
        beam: 1 or more
        top_k: default = 50
        top_p: default = 1.0
        repetition_penalty: default = 1.0
        no_repeat_ngram_size: default = 0
        length_penalty: default = 1.0
            -> Set to values < 0.0 in order to encourage the model to generate longer sequences,
                   to a value > 0.0 in order to encourage the model to produce shorter sequences.
        '''
        inputs = torch.as_tensor(instance['src_ids'], device=self.device)

        if not self.enc_dec:
            max_length = self.max_len * 2
        else:
            max_length = self.max_len

        # if self.beam > 1:
        #     outs = self.model.generate(inputs,
        #                                num_beams=self.beam,
        #                                max_length=max_length,
        #                                early_stopping=True,
        #                                length_penalty=self.length_penalty,
        #                                repetition_penalty=self.repetition_penalty,
        #                                no_repeat_ngram_size=self.no_repeat_ngram_size,
        #                                num_return_sequences=self.num_generate)
        #
        # else:
        #     outs = self.model.generate(inputs,
        #                                do_sample=True,
        #                                max_length=max_length,
        #                                top_p=self.top_p,
        #                                top_k=self.top_k,
        #                                temperature=self.temperature,
        #                                early_stopping=True,
        #                                length_penalty=self.length_penalty,
        #                                repetition_penalty=self.repetition_penalty,
        #                                no_repeat_ngram_size=self.no_repeat_ngram_size,
        #                                num_return_sequences=self.num_generate)

        outs = self.model.generate(inputs,
                                   num_beams=self.beam,
                                   max_length=max_length,
                                   early_stopping=True)

        if not self.enc_dec:
            outs = outs[:, inputs.shape[1]:]

        outs = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
        refs = self.tokenizer.batch_decode(instance['labels'], skip_special_tokens=True)
        return outs, refs


def inference_args(parent_parser=None):
    if parent_parser is not None:
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
    else:
        parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=None, type=str, help='None이면 상관x. 그외에는 csv파일 inference결과 저장할 폴더 경로를 넣어줄것')
    parser.add_argument('--testfile', default=None, type=str, help='dd')

    parser.add_argument('--max_len', type=int, default=1024, help='모델 생성 최고 길이 결정')
    parser.add_argument('--beam', type=int, default=5, help='beam size 설정')
    parser.add_argument('--top_p', type=float, default=0.92, help='top_p 비율 설정 - top_k 쓰려면 1.0으로')
    parser.add_argument('--top_k', type=int, default=0, help='top_k 설정 - top_p 쓰려면 0으로')
    parser.add_argument('--temperature', type=float, default=1.0, help='top_p 비율 설정 - top_k 쓰려면 1.0으로')
    parser.add_argument('--length_penalty', type=float, default=0.0, help='0보다 작으면 긴 문장 생성촉진')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0, help='n-gram 반복수 제한')
    parser.add_argument('--num_generate', type=int, default=1, help='생성할 문장수 결정')

    parser.add_argument('--mode', type=str, default=None, help='경로 바꾸기 귀찮아서')

    args = parser.parse_args()
    return args


