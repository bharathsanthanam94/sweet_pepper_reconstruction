import os
import subprocess
from os.path import join
import click
import torch
import yaml
from easydict import EasyDict as edict
# import sys
# sys.path.append('/home/bharath/Desktop/thesis/code/pepper_transformer/transformer-completion/transformer_completion')
from transformer_completion.datasets.fruits import IGGFruitDatasetModule
from transformer_completion.datasets.pheno4d import Pheno4DLeafDatasetModule
from transformer_completion.datasets.sbub3D import SBUB3DDatasetModule
from transformer_completion.datasets.cka_fruit import CKAFruitDatasetModule
from transformer_completion.models.mask_model import MaskPS

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


@click.command()
@click.option("--w", type=str, required=False)
@click.option("--ckpt", type=str, default=None, required=False)
@click.option("--bb_cr", type=float, default=None, required=False)
@click.option("--dec_cr", type=float, default=None, required=False)
@click.option("--iterative", is_flag=True)
@click.option("--model_cfg_path", type=str, default="../config/model.yaml", required=False)
def main(w, ckpt, bb_cr, dec_cr, iterative, model_cfg_path):
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), model_cfg_path)))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml")))
    )
    decoder_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    cfg.git_commit_version = str(
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip()
    )

    if cfg.MODEL.DATASET == "LEAVES":
        data = Pheno4DLeafDatasetModule(cfg)
    elif cfg.MODEL.DATASET == "FRUITS":
        data = IGGFruitDatasetModule(cfg)
    elif cfg.MODEL.DATASET == "CKA":
        data = CKAFruitDatasetModule(cfg)
    elif cfg.MODEL.DATASET == "SBUB":
        data = SBUB3DDatasetModule(cfg)
    else:
        raise NotImplementedError

    if bb_cr:
        cfg.BACKBONE.CR = bb_cr
    if dec_cr:
        cfg.DECODER.CR = dec_cr

    if iterative:
        cfg.DECODER.ITERATIVE_TEMPLATE = True
    model = MaskPS(cfg)
    if w:
        w = torch.load(w, map_location="cpu")
        model.load_state_dict(w["state_dict"])

    tb_logger = pl_loggers.TensorBoardLogger(
        cfg.LOGDIR + cfg.EXPERIMENT.ID, default_hp_metric=False
    )

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")

    cd_ckpt = ModelCheckpoint(
        monitor="val_chamfer_distance",
        filename=cfg.EXPERIMENT.ID +
        "_epoch{epoch:02d}_cd{val_chamfer_distance:.2f}",
        auto_insert_metric_name=False,
        mode="min",
        save_last=True,
    )

    precision_ckpt = ModelCheckpoint(
        monitor="val_precision_auc",
        filename=cfg.EXPERIMENT.ID +
        "_epoch{epoch:02d}_pr{val_precision_auc:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )

    recall_ckpt = ModelCheckpoint(
        monitor="val_recall_auc",
        filename=cfg.EXPERIMENT.ID +
        "_epoch{epoch:02d}_re{val_recall_auc:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )

    fscore_ckpt = ModelCheckpoint(
        monitor="val_fscore_auc",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_f{val_fscore_auc:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )

    trainer = Trainer(
        num_sanity_val_steps=0,
        gpus=cfg.TRAIN.N_GPUS,
        # devices=cfg.TRAIN.N_GPUS,
        accelerator="cuda",
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=[lr_monitor, fscore_ckpt,
                   precision_ckpt, recall_ckpt, cd_ckpt],
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
