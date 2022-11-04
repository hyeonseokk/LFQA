import logging


def get_logger(output_log_path=None):
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    if output_log_path is not None:
        file_handler = logging.FileHandler(output_log_path)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    if output_log_path is not None:
        logger.addHandler(file_handler)

    return logger

if __name__ == '__main__':
    print('')
    # parser = Base.add_model_specific_args(parser)
    # parser = ArgsBase.add_model_specific_args(parser)
    # parser = APEModule.add_model_specific_args(parser)
    # parser = pl.Trainer.add_argparse_args(parser)
    # args = parser.parse_args()
    # logging.info(args)
    #
    # setproctitle.setproctitle('hyeonseok {}'.format(args.save_filename.split('/')[-1]))
    # os.makedirs(os.path.join(args.save_filename, 'checkpoints'), exist_ok=True)
    # os.makedirs(os.path.join(args.save_filename, 'gen_files'), exist_ok=True)
    #
    # v_converter = VocabConverter(args.dict_file, args.sentencepiece)
    # model = APEGeneration(v_converter, args)
    # dm = APEModule(v_converter, args)
    #
    # # checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_bleu',
    # #                                                    dirpath='/home/mnt/hyeon/checkpoints/BAL_various_models/{}'.format(args.filenamesssss),
    # #                                                    filename='model_chp/{epoch:02d}-{val_loss:.3f}-{val_bleu:.2f}',
    # #                                                    verbose=True,
    # #                                                    save_last=True,
    # #                                                    mode='max',
    # #                                                    save_top_k=10,
    # #                                                    prefix='BAL')
    #
    # if args.tbpath is None:
    #     tbpath = args.save_filename
    # else:
    #     tbpath = args.tbpath
    # tb_logger = pl_loggers.TensorBoardLogger(os.path.join(tbpath, 'tb_logs'))  # args.default_root_dir
    # lr_logger = pl.callbacks.LearningRateMonitor()
    # trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, callbacks=[lr_logger],
    #                                         accumulate_grad_batches=args.accumulate_grad)  # resume_from_checkpoint=args.prev_model
    #
    # if args.prev_model is not None:
    #     print('Prev model loading')
    #     print(model.load_state_dict(torch.load(args.prev_model)['model_state_dict']))
    #
    # trainer.fit(model, dm)
    # trainer.fit(model, dm)

