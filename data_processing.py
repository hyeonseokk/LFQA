import os
import json
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_json(filename: str):
    if filename.endswith('.jsonl'):
        output = []
        with open(filename, 'r', encoding='utf-8') as f:
            for idx, line in tqdm(enumerate(f)):
                line = line.strip()
                line = json.loads(line)
                output.append(line)
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            output = json.load(f)
    return output


def write_json(obj: List[str], filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def make_csv(filename, save=False):
    def process_answer(x):
        return '@@@'.join(list(map(lambda x: x.replace('\n', ' '), x))).strip()

    def process_ctxs(x):
        return '@@@'.join(list(map(lambda x: x['title'] + ' ' + x['text'].replace('\n', ' ') + ' ', x))).strip()

    tmp = read_json(os.path.join('original', filename))
    tmp = pd.DataFrame(tmp)
    tmp['question'] = tmp['question'].apply(lambda x: x.replace('\n', ' ').strip())
    tmp['answers'] = tmp['answers'].apply(lambda x: process_answer(x))
    tmp['ctxs'] = tmp['ctxs'].apply(lambda x: process_ctxs(x))

    if save:
        tmp.to_csv(os.path.join('processed', filename.replace('.json', '.csv')), index=False)
        return 0
    return tmp


def split_csv(df, name, seed=1):
    np.random.seed(seed)
    df = df[~(df['ctxs'].isna()) & ~(df['answers'].isna())]
    dev = df.sample(n=1507)
    train = df.drop(dev.index)
    train.to_csv(os.path.join('processed', name.replace('.json', '_train.csv')), index=False)
    dev.to_csv(os.path.join('processed', name.replace('.json', '_dev.csv')), index=False)


def split_json(obj: List, name, seed=1):
    idx = np.arange(0, len(obj))
    np.random.seed(seed)
    np.random.shuffle(idx)
    n = 1507
    dev = [obj[i] for i in idx[:n]]
    train = [obj[i] for i in idx[n:]]
    write_json(dev, os.path.join('processed', name.replace('.json', '_dev.json')))
    write_json(train, os.path.join('processed', name.replace('.json', '_train.json')))


if __name__ == '__main__':
    # # csv로 만들기 -> 이걸로 하면 안될듯
    # os.makedirs('processed', exist_ok=True)
    # for f in [i for i in os.listdir('original') if i.endswith('.json')]:
    #     if f.startswith('train'):
    #         df = make_csv(f, save=False)
    #         split_csv(df, f)
    #     else:
    #         make_csv(f, save=True)

    # json으로 만들기 -> 이걸로 하면 안될듯
    os.makedirs('processed', exist_ok=True)
    for f in [i for i in os.listdir('original') if i.endswith('.json')]:
        print('now processing: ', f)
        obj = read_json(os.path.join('original', f))
        if f.startswith('train'):
            split_json(obj, f)
        else:
            write_json(obj, os.path.join('processed', f))

    # psg = os.path.join('original', 'dev_preprocessed_1507_top200_psg-top100.json')
    # phr = os.path.join('original', 'dev_preprocessed_1507_top200_phrase-top200.json')
    #
    # psg = read_json(psg)
    # phr = read_json(phr)
    #
    # psg = os.path.join('original', 'dev_preprocessed_1507_top200_psg-top100.json')
    # psg = read_json(psg)
