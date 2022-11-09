import os
from os.path import join as pjoin
import argparse
from typing import List

import pandas as pd
import numpy as np
import torch
import sacrebleu
from setproctitle import setproctitle
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchmetrics

from module.TrainingModule import PlModelModule
from module.ConfigModule import ManualArgs, config_loading, print_args
from module.ModelingModule import return_model
from module.DataModule import DataModule, LFQADataset
from module.GenerationModule import GenerationModule, find_ckpt, inference_args
from utils.training_utils import get_logger
from utils.file_utils import FileModule

fileutils = FileModule()


def manipulate_args(args, target: dict):
    for key in target:
        setattr(args, key, target[key])
    return args


def inference(parser):
    # 필요한 것 정의
    logger = get_logger()
    inference_arg = inference_args(parser)

    # 경로 정의
    CKPTPATH = "/home/mnt/hyeon/2.checkpoints/LFQA/gen_model"
    FILEPATH = "/home/mnt/hyeon/1.Dataset/LFQA/split/processed"

    if inference_arg.mode is not None:
        mappings = {'phr_phr_bart': {'ckpt': pjoin(CKPTPATH, "phr_top200/bart"),
                                     'file': pjoin(FILEPATH, "dev_preprocessed_1507_top200_phrase-top200.json")},
                    'phr_psg_bart': {'ckpt': pjoin(CKPTPATH, "psg_top10/bart"),
                                     'file': pjoin(FILEPATH, "dev_preprocessed_1507_top200_psg-top100.json")},
                    'sent_bart': {'ckpt': pjoin(CKPTPATH, "sent_top200/bart"),
                                  'file': pjoin(FILEPATH, "dev_preprocessed_1507_top200_sent-top200.json")}}
        assert inference_arg.mode in mappings
        inference_arg.ckpt = mappings[inference_arg.mode]['ckpt']
        inference_arg.testfile = mappings[inference_arg.mode]['file']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(inference_arg.ckpt)
    logger.info(inference_arg.testfile)

    ckpts = find_ckpt(pjoin(inference_arg.ckpt, 'checkpoints'))
    print_args(ckpts['args'].__dict__)
    core_model, tokenizer = return_model(ckpts['args'])

    # 실제 운용을 위한 함수 정의
    ds = LFQADataset(tokenizer=tokenizer,
                     filename=inference_arg.testfile,
                     args=ckpts['args'])
    dl = DataLoader(ds, batch_size=4, num_workers=4, shuffle=False)
    genmodule = GenerationModule(model=core_model,
                                 tokenizer=tokenizer,
                                 ckpt=ckpts,
                                 inference_args=inference_arg,
                                 device=device)

    outs, refs = [], []
    tot_len = len(dl)

    os.makedirs('results', exist_ok=True)

    with torch.no_grad():
        for idx, instance in enumerate(dl):
            # logger.info('instance')
            # logger.info(instance['src_ids'].shape)
            logger.info(f'{str(idx+1)} out of {str(tot_len)}')
            out, ref = genmodule.generate(instance)
            outs.extend(out)
            refs.extend(ref)

    # output = {str(i): {"out": o, "ref": r} for i, (o, r) in enumerate(zip(outs, refs))}
    output = [(o.replace('\n', ' '), r.replace('\n', ' ')) for o, r in zip(outs, refs)]
    output = pd.DataFrame(output)
    output.columns = ['outs', 'refs']
    output.to_csv(pjoin('results', inference_arg.testfile.split('/')[-1].replace(".jsonl", "_results.csv")))
    # fileutils.writelines(output, pjoin('results', inference_arg.testfile.split('/')[-1].replace(".json", "_results.json")))


if __name__ == '__main__':
    print(':)')
    parser = argparse.ArgumentParser()

    setproctitle('hs lfqa inference')

    inference(parser)


# if args.test_mode is None:
#     print('\n\nready!!!\n\n')
#     while True:
#         print('\n###################################\n')
#         inputs = str(input('generate : '))
#         print('\n\n')
#         outputs = model.generate(inputs)
#         for idx, text in enumerate(outputs):
#             print(idx, ': ', text, '\n')
#         print('\n###################################\n')
# else:
#     os.makedirs(args.test_mode, exist_ok=True)
#     if args.model_type == 'gpt2':
#         tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>',
#                                                             eos_token='</s>', unk_token='<unk>',
#                                                             pad_token='<pad>', mask_token='<mask>')
#     elif args.model_type == 'bart':
#         tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
#     else:
#         tokenizer = None
#
#     assert tokenizer is not None
#
#     processor = SajuDataset(tokenizer, args.test_filename, args)
#
#     data = pd.read_csv(args.test_filename)# .iloc[:10]
#     output = open(os.path.join(args.test_mode, 'results.txt'), 'w', encoding='utf-8')
#     bleu_tot, self_bleu_tot = [], []
#     for instance in tqdm(data.iloc, leave=False, desc='total: {} - now progress: '.format(str(len(data))), ncols=4):
#         input_ids, _ = processor.prepare(instance)
#         outs = model.generate(input_ids, processing=False)
#         labels = instance['label']
#         bleu, self_bleu = calc_bleu(outs, labels)
#         # print('\n##########')
#         # print('outs: ')
#         # for idx, out in enumerate(outs):
#         #     print(idx, ': ', out)
#         # print('labels: ', labels)
#         # print('bleu: ', bleu)
#         # print('self bleu: ', self_bleu)
#         # print('##########\n')
#         bleu_tot.append(bleu)
#         self_bleu_tot.append(self_bleu)
#         outs = [i.replace('\n', ' ').strip() for i in outs]
#         output.write("@@".join(outs) + '\n')
#
#     output.write('total bleu:      {}'.format(str(np.average(bleu_tot))))
#     output.write('total self_bleu: {}'.format(str(np.average(self_bleu_tot))))
#
#     output.close()
