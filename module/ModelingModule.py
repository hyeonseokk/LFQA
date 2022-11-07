'''
Model과 Tokenizer 관련 함수들을 담는 모듈
+ 입력 구성을 위해 정의해야하는 함수들
'''
import argparse
from types import SimpleNamespace
from typing import List
import re

import torch
import sentencepiece
# from fairseq.data import Dictionary
from transformers import (MBartForConditionalGeneration,
                          PreTrainedTokenizerFast,
                          GPT2LMHeadModel,
                          BartForConditionalGeneration,
                          AutoConfig,
                          AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LongformerForQuestionAnswering,
                          ReformerModelWithLMHead,
                          ReformerTokenizer)

from utils.training_utils import get_logger

parser = argparse.ArgumentParser()
logger = get_logger()
# logger.setLevel(logging.INFO)
torch.set_num_threads(1)


def return_sp(sp_path):
    '''
    spm_train --input=all.txt --model_prefix=a --vocab_size=50000
    --model_type=unigram --pad_id 0 --bos_id 1 --unk_id 2 --eos_id 3
    '''
    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(sp_path)
    sp._add_eos = True  # eos 붙여줘야함...
    return sp


def return_model(args: SimpleNamespace):
    pdict = {'pegasus': 'google/pegasus-xsum',
             'reformer2': 'google/reformer-crime-and-punishment',
             'reformer': "google/reformer-enwik8",
             'bart': 'facebook/bart-large-cnn',
             'longt52': 'Stancld/longt5-tglobal-large-16384-pubmed-3k_steps',
             'longt53': 'pszemraj/long-t5-tglobal-base-16384-book-summary',
             'longt5': 'google/long-t5-tglobal-base'}

    assert args.model_type in pdict  # 'Model type defining error'
    model_type = pdict[args.model_type]
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    # 일반적인 상황
    if 'reformer' in args.model_type:
        model_function = AutoModelForCausalLM
    else:
        model_function = AutoModelForSeq2SeqLM

    # 각 케이스별 특수상황
    if args.model_type == 'reformer':
        model_function = ReformerModelWithLMHead
        tokenizer = ReformerTokenizer.from_pretrained(model_type)
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token

    elif args.model_type == 'longt5':
        tokenizer.bos_token = tokenizer.pad_token


    if args.precision == 16:
        core_model = model_function.from_pretrained(model_type, torch_dtype=torch.float16)
    else:
        core_model = model_function.from_pretrained(model_type)

    return core_model, tokenizer


def replace_unicode_punct(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    """
    text = str(text)
    text = text.replace("，", ",")
    text = re.sub(r"。\s*", ". ", text)
    text = text.replace("、", ",")
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    text = text.replace("∶", ":")
    text = text.replace("：", ":")
    text = text.replace("？", "?")
    text = text.replace("《", '"')
    text = text.replace("》", '"')
    text = text.replace("）", ")")
    text = text.replace("！", "!")
    text = text.replace("（", "(")
    text = text.replace("；", ";")
    text = text.replace("１", "1")
    text = text.replace("」", '"')
    text = text.replace("「", '"')
    text = text.replace("０", "0")
    text = text.replace("３", "3")
    text = text.replace("２", "2")
    text = text.replace("５", "5")
    text = text.replace("６", "6")
    text = text.replace("９", "9")
    text = text.replace("７", "7")
    text = text.replace("８", "8")
    text = text.replace("４", "4")
    text = re.sub(r"．\s*", ". ", text)
    text = text.replace("～", "~")
    text = text.replace("’", "'")
    text = text.replace("…", "...")
    text = text.replace("━", "-")
    text = text.replace("〈", "<")
    text = text.replace("〉", ">")
    text = text.replace("【", "[")
    text = text.replace("】", "]")
    text = text.replace("％", "%")
    return text


def characterize(target: List[int], tokenizer, ref=False, max_len=512):
    '''
    :param target: 정수 시퀀스 -> tokenizer로 디코딩해야함. 한문장 한문장 하는 것을 기준으로 하기
    :param tokenizer: 이 함수는 fairseq을 huggingface로 옮길때 주로 활용될 것 -> sentencepiece모델
    :param ref: True라면 단순 pad 날리기, False라면 EOS찾아서 그 뒤를 날리기
    :return:
    '''
    if ref:
        skips = [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]
        return ''.join(tokenizer.decode([k for k in target if k not in skips])).replace('▁', ' ')
    else:  # 모델이 생성한 결과물을 복호화하기
        # candi에서는 eos이후에 오는애를 전부 날려야
        eos = target.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in target else (max_len - 1) # 원래 이거였음
        target = target[1:eos]  # bos 제거하기 위함
        return ''.join(tokenizer.decode(target)).replace('▁', ' ')


# def return_model(huggingface_model_ft, huggingface_model_name, precision):
#     # precision 16 / 32때문에 만들은 것
#     if precision == 32:
#         return huggingface_model_ft.from_pretrained(huggingface_model_name)
#     elif precision == 16:
#         model = huggingface_model_ft(AutoConfig.from_pretrained(huggingface_model_name))
#
#         return 0
#     else:
#         raise Exception('huggingface model definition error')


