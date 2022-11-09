import os
import argparse
from typing import List
import json

import pandas as pd
import numpy as np
import torch
import sacrebleu
from setproctitle import setproctitle
from transformers import PreTrainedTokenizerFast, AutoModel, AutoTokenizer
from utils.config_utils import print_args
from tqdm import tqdm
from KoBERTScore import BERTScore

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data = ['bart_split1_c1', 'bart_split1_c1c2', 'bart_split2_c1', 'bart_split2_c1c2',
#         'gpt2_split1_c1', 'gpt2_split1_c1c2', 'gpt2_split2_c1', 'gpt2_split2_c1c2']

# data = ['bart_split1_c1_topp', 'bart_split2_c1_topp', 'gpt2_split1_c1_topp', 'gpt2_split2_c1_topp']
# data = ['bart_split1_c1_topp', 'bart_split2_c1_topp']
# data = [os.path.join('results', i, 'results.txt') for i in data]


def calc_bleu(hyp: List[str], ref: str):
    # hyp는 model output.. 20개씩 출력한것
    # ref는 문장 한개임

    # BLEU
    bleu = sacrebleu.corpus_bleu(hyp, [[ref for _ in range(len(hyp))]], tokenize='ko-mecab').score

    # self-BLEU
    self_ref = [[i for i in hyp[:hyp.index(h)] + hyp[hyp.index(h)+1:]] for h in hyp]
    self_hyp = [[h]*(len(hyp) - 1) for h in hyp]
    self_bleu = []

    for h, r in zip(self_hyp, self_ref):
        self_bleu.append(sacrebleu.corpus_bleu(h, [r], tokenize='ko-mecab').score)

    self_bleu = np.average(self_bleu)
    return bleu, self_bleu


def calc_bertscore(hyp: List[str], ref: str, bert):
    return bert.score(hyp, [ref]*len(hyp), batch_size=128, verbose=False)


def calc_precision(hyp: List[str], nouns: str):
    nouns = nouns.split('#')
    output = 0
    for h in hyp:
        cnt = 0
        for noun in nouns:
            cnt += (noun in h) * 1
        output += (cnt / len(nouns))
    return output / len(hyp)


def calc_score(filename, bert):
    with open(filename, 'r', encoding='utf-8') as f:
        results = f.readlines()
    results = [i.replace('\n', ' ').strip() for i in results][:-1]

    if 'split1' in filename:
        df = pd.read_csv('../data/Data_processed/Data_split1_processed_test.csv')
    else:
        df = pd.read_csv('../data/Data_processed/Data_split2_processed_test.csv')

    labels = df['label'].tolist()
    nouns = df['input_noun'].tolist()

    output = {'bleu': [],
              'self_bleu': [],
              'bert_score': [],
              'precision': []}
    for idx, (hyp, ref, noun) in enumerate(zip(results, labels, nouns)):
        hyp = hyp.split('@@')
        bleu, self_bleu = calc_bleu(hyp, ref)
        bert_score = calc_bertscore(hyp, ref, bert)
        precision = calc_precision(hyp, noun)
        output['bleu'].append(bleu)
        output['self_bleu'].append(self_bleu)
        output['bert_score'].append(bert_score)
        output['precision'].append(precision)
        # print(output)

    for key in output:
        output[key] = np.average(output[key])
    return output


def load_bert(modelname):
    # tokenizer = AutoTokenizer.from_pretrained(modelname)
    # model = AutoModel.from_pretrained(modelname)
    bert = BERTScore(modelname)
    return bert

if __name__ == '__main__':

    bert = load_bert('beomi/kcbert-base')
    output = {}
    for d in data:
        print(d)
        output[d] = calc_score(d, bert)

    with open('analyze_output.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)
