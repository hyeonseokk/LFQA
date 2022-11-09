'''
데이터 다루는 것과 관련한 모듈들 정리
'''
import os
import sys
import logging
import copy
import argparse
from os.path import join as pjoin
from pathlib import Path

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_utils import TorchFileModule
from utils.training_utils import get_logger
from module.ModelingModule import replace_unicode_punct

logger = get_logger()

class SajuDataset(Dataset):
    '''
    spm_train --input=all.txt --model_prefix=a --vocab_size=50000 --model_type=unigram --pad_id 1 --bos_id 0 --unk_id 3 --eos_id 2
    args는 training에 넣어주는 args가 들어올 것임
    '''
    def __init__(self, tokenizer, filename, args):
        super().__init__()
        self.args = args
        self.filename = filename
        self.tokenizer = tokenizer  # tokenizer는 사용하는 모델에 따라서 정의한 이후, 여기에 넣어주면 될듯
        self._declare()
        self.fileutils = TorchFileModule()

        if not Path(self.args.cache_dir).is_dir():
            os.makedirs(self.args.cache_dir, exist_ok=True)

        if self.use_cache:  # 속도 향상을 위해 캐싱 진행하는 것
            if self.cache_filename not in os.listdir(self.args.cache_dir):
                self._caching()
            self._read_cache()
        else:  # 캐싱하지 않고 get_item에서 입력 구성하는 경우
            self.docs = self.fileutils.reads(filename)  # 여기서는 csv쓸것
            self.len = len(self.docs)

    def _declare(self):
        if 'gpt' in self.args.model_type:
            self.enc_dec = False
        else:
            self.enc_dec = True
        self.max_len = self.args.max_len
        self.input_type = self.args.input_type
        self.cache_filename = self.filename.split('/')[-1] + '.cache'
        self.use_cache = self.args.use_cache

        self.aux = False
        if self.args.training_type == 'type2':
            self.aux = True

        self.pad_index = self.tokenizer.pad_token_id
        self.eos_index = self.tokenizer.eos_token_id
        self.bos_index = self.tokenizer.bos_token_id
        # self.tokenizer.pad_token = '<pad>'

    def _caching(self):
        docs = self.fileutils.reads(self.filename)  # 여기서는 csv쓸것
        self.len = len(docs)
        sample = list(self.get_from_doc(docs.iloc[0]).keys())  # output이 뭘 담아야 하는지 적혀있음

        os.makedirs(pjoin(self.args.cache_dir, self.cache_filename), exist_ok=True)
        cached_docs = {key: np.memmap(pjoin(self.args.cache_dir, self.cache_filename, key + '.npy'),
                                      mode='w+', dtype=np.int_, shape=(self.len, self.max_len))
                       for key in sample}

        for idx in tqdm(range(len(docs)), desc='now data caching ...'):
            processed = self.get_from_doc(docs.iloc[idx])
            for key in processed:
                cached_docs[key][idx, :] = processed[key]

    def _read_cache(self):
        docs = self.fileutils.reads(self.filename)  # 여기서는 csv쓸것
        self.len = len(docs)
        sample = list(self.get_from_doc(docs.iloc[0]).keys())  # output이 뭘 담아야 하는지 적혀있음

        self.docs = {key: np.memmap(pjoin(self.args.cache_dir, self.cache_filename, key + '.npy'),
                                    mode='r+', dtype=np.int_, shape=(self.len, self.max_len))
                     for key in sample}

    def add_padding_data(self, inputs: np.ndarray, left=True):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            if left:
                inputs = np.concatenate([pad, inputs])
            else:
                inputs = np.concatenate([inputs, pad])
        inputs = inputs[:self.max_len]
        return inputs
    
    def prepare_input(self, instance):
        # padding하기 이전, input, label을 정의해주기 -> eos, bos같은것 다 붙인 상태로
        temp = {'categoty1': self.tokenizer.encode(replace_unicode_punct(instance['input_category1'])),
                'categoty2': self.tokenizer.encode(replace_unicode_punct(instance['input_category2'])),
                'categoty3': self.tokenizer.encode(replace_unicode_punct(instance['input_category3'])),
                'noun': self.tokenizer.encode(replace_unicode_punct(instance['input_noun']))}
        if self.aux:
            temp.update({'aux': self.tokenizer.encode(replace_unicode_punct(instance['input_aux']))})

        input_ids = [self.bos_index]
        if 'c1' in self.input_type:
            input_ids = input_ids + temp['categoty1'] + [self.eos_index]
        if 'c2' in self.input_type:
            input_ids = input_ids + temp['categoty2'] + [self.eos_index]
        if 'c3' in self.input_type:
            input_ids = input_ids + temp['categoty3'] + [self.eos_index]
        input_ids = input_ids + temp['noun'] + [self.eos_index]

        if self.aux:
            input_ids = input_ids + temp['aux'] + [self.eos_index]

        return input_ids

    def prepare_label(self, instance):
        # padding하기 이전, input, label을 정의해주기 -> eos, bos같은것 다 붙인 상태로
        return self.tokenizer.encode(replace_unicode_punct(instance['label'])) + [self.eos_index]

    def ready_(self, input_ids, label_ids):
        # 여기에서 구성해주는 output의 key에 따라 캐싱할 파일들의 개수 등이 결정됨 - 웬만하면 거의 안바뀌고 이대로 갈 듯??
        if self.enc_dec:
            # BART등의 enc-dec모델을 위함
            dec_input_ids = [self.bos_index]
            dec_input_ids += label_ids[:-1]
            dec_input_ids = self.add_padding_data(dec_input_ids, left=False)
            input_ids = self.add_padding_data(input_ids, left=False)
            label_ids = self.add_padding_data(label_ids, left=False)
            output = {'src_ids': np.array(input_ids, dtype=np.int_),
                      'labels': np.array(label_ids, dtype=np.int_),
                      'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                      'label_length': np.array([0], dtype=np.int_)}
        else:
            # GPT2등의 decoder모델을 위함
            label_length = len(label_ids)
            label_ids = copy.deepcopy(input_ids + label_ids)
            label_ids = self.add_padding_data(label_ids, left=True)
            label_length = len(label_ids) - label_length
            inference_ids = copy.deepcopy(input_ids)
            inference_ids = self.add_padding_data(inference_ids, left=True)
            input_ids = copy.deepcopy(label_ids)
            output = {'src_ids': np.array(input_ids, dtype=np.int_),
                      'labels': np.array(label_ids, dtype=np.int_),
                      'inference_ids': np.array(inference_ids, dtype=np.int_),
                      'label_length': np.array([label_length], dtype=np.int_)}
        return output

    def get_from_doc(self, instance):
        input_ids = self.prepare_input(instance)
        label_ids = self.prepare_label(instance)
        output = self.ready_(input_ids, label_ids)
        return output

    def get_from_cache(self, idx):
        output = {}
        for key in self.docs:
            output[key] = self.docs[key][idx, :]
        return output

    def __getitem__(self, idx):
        if self.use_cache:
            return self.get_from_cache(idx)
        else:
            return self.get_from_doc(self.docs.iloc[idx])

    def __len__(self):
        return self.len


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, args):
        super().__init__()
        self.tokenizer = tokenizer
        self.hparam_args = args
        self.batch_size = args.batch_size
        self.max_len = args.max_len
        self.train_file = args.train_file
        self.valid_file = args.valid_file
        self.num_workers = args.num_workers

        self.train = SajuDataset(self.tokenizer, self.train_file, self.hparam_args)
        self.test = SajuDataset(self.tokenizer, self.valid_file, self.hparam_args)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        return parser

    def train_dataloader(self):
        train = DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return val


