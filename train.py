import argparse
import logging
import os

from setproctitle import setproctitle
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import torch

from module.TrainingModule import PlModelModule
from module.ConfigModule import ManualArgs, config_loading, print_args
from module.ModelingModule import return_model
from module.DataModule import DataModule
from utils.training_utils import get_logger

logger = get_logger()


def train(parser):
    parser = ManualArgs().make_manual_config(save=None, parser=parser)
    args = parser.parse_args()
    args = config_loading(parser, config_manual=args.config_manual, config_trainer=args.config_trainer)

    args.gradient_clip_val = 1.0
    args.default_root_dir = 'logs'
    args.gpus = 1

    print_args(args.__dict__, 'logging')

    if args.proctitle is None:
        setproctitle('hs lfqa {}'.format(args.model_type))
    else:
        setproctitle(args.proctitle)

    os.makedirs(os.path.join(args.save_filename, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.save_filename, 'gen_files'), exist_ok=True)

    '''
    - core_model.resize_token_embedding(new_embedding_size) 이걸로 model embedding size 조절
    - tokenizer.add_tokens(['<홍길동>', '<홍길순>']) 과 같은 것으로 vocab size 조절 + 새로운 embedding token 추가
    아래 코드는 training utils로 빼서, tokenizer / model 불러오는 코드 새롭게 짜는게 나을듯함
    '''
    core_model, tokenizer = return_model(args)

    dm = DataModule(tokenizer, args)
    model = PlModelModule(tokenizer, core_model, args)

    if args.tbpath is None:
        tbpath = args.save_filename
    else:
        tbpath = args.tbpath
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(tbpath, 'tb_logs'))  # args.default_root_dir
    lr_logger = pl.callbacks.LearningRateMonitor()

    # plugins = DeepSpeedPlugin(stage=3,
    #                           offload_optimizer=True,
    #                           offload_parameters=True,
    #                           cpu_offload=True)
    args.gpus = 4
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=tb_logger, callbacks=[lr_logger],
                                            accumulate_grad_batches=args.accumulate_grad,
                                            plugins=args.plugins,
                                            precision=args.precision)  # resume_from_checkpoint=args.prev_model

    if args.prev_model is not None:
        print('Prev model loading')
        print(model.load_state_dict(torch.load(args.prev_model)['model_state_dict']))

    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    train(parser)

