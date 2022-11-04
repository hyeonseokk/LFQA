'''
pytorch lightning을 사용함에 있어, 훈련에 필요한 사항들을 정리해주는 함수들을 담음
'''
import argparse
import logging
import os
import sys
from collections import OrderedDict
import yaml

import setproctitle
import sacrebleu
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from utils.training_utils import get_logger
from utils.file_utils import TorchFileModule

parser = argparse.ArgumentParser()
logger = get_logger()
# logger.setLevel(logging.INFO)
torch.set_num_threads(1)


class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.hparam_args = hparams
        self.beam_size = hparams.beam_size
        self.max_len = hparams.max_len
        self.fileutils = TorchFileModule()
        self.ckpt_dir = os.path.join(hparams.save_filename, 'checkpoints')
        self.gen_dir = os.path.join(hparams.save_filename, 'gen_files')
        self.ckpt_save_num = hparams.ckpt_save_num
        self.bos_token = '<s>'
        self.eos_token = '</s>'

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if self.hparam_args.optimizer == 'AdamW':
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparam_args.lr)
        elif self.hparam_args.optimizer == 'FusedAdam':
            optimizer = FusedAdam(optimizer_grouped_parameters, lr=self.hparam_args.lr)
        elif self.hparam_args.optimizer == 'DeepSpeedCPUAdam':
            optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=self.hparam_args.lr)
        else:
            raise Exception('optimizer setting error')

        # warm up lr
        num_workers = (self.hparam_args.gpus if self.hparam_args.gpus is not None else 1) * (
            self.hparam_args.num_nodes if self.hparam_args.num_nodes is not None else 1)
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        if self.hparam_args.max_epochs is not None:
            logger.info(data_len)
            logger.info(self.hparam_args.batch_size)
            logger.info(num_workers)
            num_train_steps = int(data_len / (self.hparam_args.batch_size * num_workers) * self.hparam_args.max_epochs)
        else:
            num_train_steps = self.hparam_args.max_steps
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparam_args.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


class PlModelModule(Base):
    def __init__(self, tokenizer, model, hparams, **kwargs):
        super(PlModelModule, self).__init__(hparams, **kwargs)

        if 'gpt' in hparams.model_type:
            self.enc_dec = False
            self.valid_maxlen = self.max_len * 2
        else:  # 아마 다 이쪽일것임
            self.enc_dec = True
            self.valid_maxlen = self.max_len

        self.tokenizer = tokenizer
        self.model = model

        self.model.train()  # 이걸로 dropout activate

        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

    def forward(self, inputs):
        if self.enc_dec:
            src_attention_mask = inputs['src_ids'].ne(self.pad_token_id).float()
            decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
            return self.model(input_ids=inputs['src_ids'],
                              attention_mask=src_attention_mask,
                              decoder_input_ids=inputs['decoder_input_ids'],
                              decoder_attention_mask=decoder_attention_mask,
                              labels=inputs['labels'], return_dict=True)
        else:
            src_attention_mask = inputs['src_ids'].ne(self.pad_token_id).float()
            return self.model(input_ids=inputs['src_ids'],
                              attention_mask=src_attention_mask,
                              labels=inputs['labels'], return_dict=True)

    def training_step(self, batch, batch_idx):
        self.model.train()
        outs = self(batch)
        # loss = self.loss_fn(outs['logits'].view(-1, self.model.config.vocab_size), batch['labels'].view(-1))
        loss = outs['loss']

        self.log('train_loss', loss, prog_bar=True)
        self.log('now_step', self.global_step, prog_bar=True)
        # if self.global_step in self.hparam_args.ckpt_freq:
        #     self.save_middle_ckpt(loss)
        return loss

    def save_middle_ckpt(self, loss):
        self.fileutils.save_one(self, loss, 0, filename=os.path.join(self.ckpt_dir, 'step{}.pt'.format(str(self.global_step))))

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        if self.enc_dec:
            input_temp = batch['src_ids']
        else:
            input_temp = batch['inference_ids']
        
        outs = self.model.generate(input_temp, num_beams=self.beam_size, max_length=self.valid_maxlen, early_stopping=True)

        num = batch['labels'].shape[0]

        model_input = self.tokenizer.batch_decode(input_temp, skip_special_tokens=False)
        
        if self.enc_dec:  # BART등의 모델인 경우
            candi = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
            ref = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        else:  # GPT2등의 모델인 경우
            candi, ref = [], []
            for idx in range(outs.shape[0]):  # batch size
                candi.append(self.tokenizer.decode(outs[idx][input_temp.shape[1]:], skip_special_tokens=True))
                ref.append(self.tokenizer.decode(batch['labels'][idx][batch['label_length'][idx]:], skip_special_tokens=True))        
        # candi, ref = self.characterize(batch, outs)

        loss_out = self(batch)
        loss = self.loss_fn(loss_out['logits'].view(-1, self.model.config.vocab_size),
                            batch['labels'].view(-1)) * num
        return (loss, num, candi, ref, model_input)

    def validation_epoch_end(self, outputs):
        losses = []
        candis, refs, model_inputs = [], [], []
        tot = 0
        for loss, num, candi, ref, model_input in outputs:
            losses.append(loss.cpu())
            candis.extend(candi)
            refs.extend(ref)
            model_inputs.extend(model_input)
            tot += num
        print('current steps: ', self.global_step)
        loss = torch.sum(torch.tensor(losses, dtype=torch.float)) / tot
        bleu = sacrebleu.corpus_bleu(candis, [refs]).score

        if self.current_epoch == 0:
            self.fileutils.writelines(refs, os.path.join(self.gen_dir, 'ref.ref'))
            self.fileutils.writelines(model_inputs, os.path.join(self.gen_dir, 'inp.inp'))

        candi_filename = f'epoch{format(self.current_epoch, "03")}_bleu{format(bleu, ".4f")}.candi'
        self.fileutils.writelines(candis, os.path.join(self.gen_dir, candi_filename))

        self.fileutils.ckpt_save(plself=self,
                                 loss=loss,
                                 score=bleu)

        self.log('val_bleu', bleu, prog_bar=True)



