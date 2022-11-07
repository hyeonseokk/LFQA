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
from module.ModelingModule import return_model, replace_unicode_punct, characterize

logger = get_logger()
# logger.setLevel(logging.INFO)
torch.set_num_threads(1)


class GenerationModule:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.fileutils = TorchFileModule()
        self._get_variables()
        self._get_model_tokenizer()
        self._apply_ckpt()

    def _get_variables(self):
        # 필요한 변수들 args에서 가져오기
        self.ckpt = self.args.ckpt

        self.num_workers = self.args.num_workers
        self.max_len = self.args.max_len
        self.beam = self.args.beam
        self.top_p = self.args.top_p
        self.top_k = self.args.top_k
        self.temperature = self.args.temperature
        self.length_penalty = self.args.length_penalty
        self.repetition_penalty = self.args.repetition_penalty
        self.no_repeat_ngram_size = self.args.no_repeat_ngram_size
        self.num_generate = self.args.num_generate

    def _get_model_tokenizer(self):
        if 'gpt' in self.args.model_type:
            self.enc_dec = False
        else:
            self.enc_dec = True

        self.model, self.tokenizer = return_model(self.args)

        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.model.eval()

    def _apply_ckpt(self):
        # ckpt경로에서부터 best 체크포인트를 찾아 적용하는 것
        ckpt = None
        if self.ckpt is not None:
            if self.ckpt.endswith('.pt'):
                ckpt = self.ckpt
            else:  # 입력으로 폴더 경로가 주어지는 경우
                ckpt = None
                for item in os.listdir(self.ckpt):
                    if 'best' in item:
                        if item.endswith('.pt'):
                            ckpt = item
                            break
                if (ckpt is None) and (os.listdir(self.ckpt) != []):
                    ckpt = os.listdir(self.ckpt)[0]

        if ckpt is not None:
            self.model.load_state_dict(torch.load(os.path.join(self.ckpt, ckpt))['model_state_dict'])
        else:
            print('\n\n No checkpoint has been loaded! \n\n')
            logger.info('\n\n No checkpoint has been loaded! \n\n')

        self.model = self.model.to(self.device)

    def preprocessing(self, sentence: Union[str, np.ndarray], processing=True):
        # string형식의 문장 하나 받음 -> 모델 입력 가능한 tensor형태로 출력
        if processing:
            output = replace_unicode_punct(sentence)
            output = self.tokenizer.encode(output)
        else:
            output = sentence

        if len(output) < self.max_len:
            pad = np.array([self.pad_token_id] * (self.max_len - len(output)))
            if self.enc_dec:
                output = np.concatenate([output, pad])
            else:
                output = np.concatenate([pad, output])
                
        output = output[:self.max_len]
        output = np.expand_dims(output, 0)
        return torch.tensor(output, dtype=torch.int32, device=self.device)

    @torch.no_grad()
    def generate(self, inputs: str, processing=True):
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
        inputs = self.preprocessing(inputs, processing)

        if not self.enc_dec:
            max_length = self.max_len * 2
        else:
            max_length = self.max_len
        if self.beam > 1:
            outs = self.model.generate(inputs,
                                       num_beams= self.beam,
                                       max_length=max_length,
                                       early_stopping=True,
                                       length_penalty=self.length_penalty,
                                       repetition_penalty=self.repetition_penalty,
                                       no_repeat_ngram_size=self.no_repeat_ngram_size,
                                       num_return_sequences=self.num_generate)
        else:
            outs = self.model.generate(inputs,
                                       do_sample=True,
                                       max_length=max_length,
                                       top_p=self.top_p,
                                       top_k=self.top_k,
                                       temperature=self.temperature,
                                       early_stopping=True,
                                       length_penalty=self.length_penalty,
                                       repetition_penalty=self.repetition_penalty,
                                       no_repeat_ngram_size=self.no_repeat_ngram_size,
                                       num_return_sequences=self.num_generate)

        if not self.enc_dec:
            outs = outs[:, inputs.shape[1]:]
        # print(outs)

        if self.args.model_type == 'fairseq_manual':
            outs = characterize(outs, self.tokenizer)
        else:
            outs = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
        return outs


def inference_args(parent_parser=None):
    if parent_parser is not None:
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
    else:
        parser = argparse.ArgumentParser()
    # 파일 불러오기 관련
    parser.add_argument('--ckpt', type=str, default=None, help='체크포인트 경로 - .pt파일 혹은 폴더명')
    parser.add_argument('--test_filename', type=str, default=None, help='테스트할 tsv파일. None이면 sentence별 작동')

    # 모델 생성 관련
    parser.add_argument('--max_len', type=int, default=128, help='모델 생성 최고 길이 결정')
    parser.add_argument('--beam', type=int, default=1, help='beam size 설정')
    parser.add_argument('--top_p', type=float, default=0.92, help='top_p 비율 설정 - top_k 쓰려면 1.0으로')
    parser.add_argument('--top_k', type=int, default=0, help='top_k 설정 - top_p 쓰려면 0으로')
    parser.add_argument('--temperature', type=float, default=1.0, help='top_p 비율 설정 - top_k 쓰려면 1.0으로')
    parser.add_argument('--length_penalty', type=float, default=0.0, help='0보다 작으면 긴 문장 생성촉진')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0, help='n-gram 반복수 제한')
    parser.add_argument('--num_generate', type=int, default=20, help='생성할 문장수 결정')

    # 기타 정의 필요한 부분
    parser.add_argument('--model_type', type=str, default='bart', choices=['pegasus', 'reformer', 'longformer', 'bart'])
    parser.add_argument('--input_type', type=str, default='c1',help='c1, c2, c3 을 넣어주기..')
    parser.add_argument('--num_workers', type=int, default=8, help='test_filename is not None인 경우에만 필요')

    # 불필요한 부분 - 현재 미구현 상태
    parser.add_argument('--path_sp', type=str, default="", help='sentencepiece 모델 경로 - 현재 미구현')
    parser.add_argument('--path_fd', type=str, default="" ,help='fairseq dict 경로 - 현재 미구현')

    args = parser.parse_args()
    return args


