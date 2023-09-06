import os
import subprocess
from os.path import join

import click
import torch
import yaml
from easydict import EasyDict as edict
from transformer_completion.datasets.fruits import IGGFruitPretrainDatasetModule
from transformer_completion.datasets.pheno4d import Pheno4DLeafDatasetModule
from transformer_completion.models.pre_backbone import PreBackbone
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


@click.command()
@click.option("--w", type=str, default=None, required=False)
@click.option("--ckpt", type=str, default=None, required=False)
@click.option("--bb_cr", type=float, default=None, required=False)
@click.option("--iterative", is_flag=True)
def main(w, ckpt, bb_cr, iterative):
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__),
                                 "../config/backbone.yaml"))))
    decoder_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    cfg.git_commit_version = str(
        subprocess.check_output(["git", "rev-parse", "--short",
                                 "HEAD"]).strip())

    if cfg.MODEL.DATASET == 'LEAVES':
        data = Pheno4DLeafDatasetModule(cfg)
    elif cfg.MODEL.DATASET == 'FRUITS':
        data = IGGFruitPretrainDatasetModule(cfg)
    else:
        raise NotImplementedError

    if bb_cr:
        cfg.BACKBONE.CR = bb_cr
    if iterative:
        cfg.DECODER.ITERATIVE_TEMPLATE = True
    model = PreBackbone(cfg)
    if w:
        w = torch.load(w, map_location="cpu")
        model.load_state_dict(w["state_dict"])

    tb_logger = pl_loggers.TensorBoardLogger(cfg.LOGDIR + cfg.EXPERIMENT.ID,
                                             default_hp_metric=False)

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        num_sanity_val_steps=0,
        gpus=cfg.TRAIN.N_GPUS,
        accelerator="cuda",
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=[lr_monitor],  # , pq_ckpt, iou_ckpt],
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        resume_from_checkpoint=ckpt,
    )

    trainer.fit(model, data)
    trainer.test(model, dataloaders=data.test_dataloader())


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


if __name__ == "__main__":
    main()
