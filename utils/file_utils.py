import os
import json
import argparse
import yaml
from collections import OrderedDict
from typing import List

from tqdm import tqdm
import pandas as pd
import torch


class FileModule:
    @staticmethod
    def read_file(filename: str):
        with open(filename, 'r', encoding='utf-8') as f:
            tmp = f.readlines()
        return [i[:-1] for i in tmp]

    @staticmethod
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

    @staticmethod
    def read_csv(filename: str):
        if filename.endswith('.tsv'):
            return pd.read_csv(filename, sep='\t')
        else:
            return pd.read_csv(filename)

    @staticmethod
    def writelines(obj: List[str], filename):
        obj = list(map(lambda i: str(i).strip() + '\n', obj))
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(obj)

    @staticmethod
    def writejson(obj: List[str], filename):
        if filename.endswith('.jsonl'):
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(obj, f)
                f.write('\n')
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(obj, f)

    def reads(self, filename):
        if filename.endswith('.json') or filename.endswith('.jsonl'):
            return self.read_json(filename)
        elif filename.endswith('.tsv') or filename.endswith('.csv'):
            return self.read_csv(filename)
        return self.read_file(filename)


class TorchFileModule(FileModule):
    @staticmethod
    def find_best(dirname):
        out = None
        for files in os.listdir(dirname):
            if files.startswith('best_'):
                out = files
        return out

    @staticmethod
    def import_weight(old_state: str):
        # fairseq것을 받아서 huggingface로 바꿔주는것..
        # translation = self.hparam_args.translation_model_ckpt
        translation_state_dict = torch.load(old_state)['model']

        new_state_dict = OrderedDict()
        new_state_dict.update({'model.shared.weight': translation_state_dict['encoder.embed_tokens.weight']})

        for item in translation_state_dict:
            new_state_dict.update({'model.' + item: translation_state_dict[item]})
        new_state_dict.update({'lm_head.weight': translation_state_dict['decoder.output_projection.weight']})
        return new_state_dict

    @staticmethod
    def save_one(plself, loss, bleu, filename):
        # self: pl module내에서 실제 self를 인자로 넣어주면됨
        torch.save({'epoch': plself.current_epoch,
                    'step': plself.global_step,
                    'model_state_dict': plself.model.state_dict(),
                    'optimizer_state_dict': plself.optimizers().state_dict(),
                    'loss': loss,
                    'sacrebleu': bleu,
                    'args': plself.hparam_args},
                   filename)

    def ckpt_save(self, plself, loss, score, maximize=True, score_name=None, step=False, last=False):
        '''
        :param plself: pl module내에서 실제 self를 인자로 넣어주면됨
        :param loss: loss를 입력으로
        :param score: 목표로 하는 score
        :param maximize: score를 maximize해서 저장할지 minimize해서 저장할지
        :param step: True면 step을 기준으로 저장, False면 epoch이름을 기준으로 저장
        :return:
        '''
        names = {'loss': str(format(loss, '.3f')),
                 'score': str(format(score, '.3f')),
                 'epoch': format(plself.current_epoch, '03'),
                 'step': format(plself.global_step, '07')}

        if score_name is None:
            score_name = 'score'

        if step:
            iteration_name = f'step={names["step"]}_{score_name}={names["score"]}.pt'
            best_name = f'best_{score_name}={names["score"]}_step={names["step"]}.pt'
        else:
            iteration_name = f'epoch={names["epoch"]}_{score_name}={names["score"]}.pt'
            best_name = f'best_{score_name}={names["score"]}_epoch={names["epoch"]}.pt'

        iteration_name = os.path.join(plself.ckpt_dir, iteration_name)
        best_name = os.path.join(plself.ckpt_dir, best_name)

        save_list = [i for i in os.listdir(plself.ckpt_dir) if ('.pt' in i) and (score_name in i)]
        ################################### 중간과정 결과물 저장 #####################################
        if len(save_list) < plself.ckpt_save_num:
            self.save_one(plself, loss, score, filename=iteration_name)
        else:
            bleu_dict = {i: float((i.split('.pt')[0].split(f'_{score_name}=')[1])) for i in save_list if
                         ('best' not in i)}
            if maximize:
                minval = min(bleu_dict.values())
                if minval < score:
                    for fn in bleu_dict:
                        if bleu_dict[fn] == minval:
                            if os.path.exists(os.path.join(plself.ckpt_dir, fn)):
                                os.remove(os.path.join(plself.ckpt_dir, fn))
                            break
                    self.save_one(plself, loss, score, filename=iteration_name)
            else:  # minimize
                maxval = max(bleu_dict.values())
                if maxval > score:
                    for fn in bleu_dict:
                        if bleu_dict[fn] == maxval:
                            if os.path.exists(os.path.join(plself.ckpt_dir, fn)):
                                os.remove(os.path.join(plself.ckpt_dir, fn))
                            break
                    self.save_one(plself, loss, score, filename=iteration_name)

        ################################### best score 저장 #####################################
        best_bleu = None
        for i in save_list:
            if 'best' in i:
                best_bleu = i
                break

        if best_bleu is not None:
            best_bleu_score = float(best_bleu.split('_')[1].split('=')[1])
            if best_bleu_score < score:
                if os.path.exists(os.path.join(plself.ckpt_dir, best_bleu)):
                    os.remove(os.path.join(plself.ckpt_dir, best_bleu))
                self.save_one(plself, loss, score, filename=best_name)
        else:
            self.save_one(plself, loss, score, filename=best_name)

        ################################### last model 저장 #####################################
        if last:
            self.save_one(plself, loss, score, filename=os.path.join(plself.ckpt_dir, 'model_last.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=None, type=str)
    args = parser.parse_args()
