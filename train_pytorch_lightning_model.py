import glob
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

import wandb
from dtl_model import DtlModel
from loki_datasets import LokiDataModule, LokiTrainValDataset

wandb.login()

torch.manual_seed(42)
seed = seed_everything(42, workers=True)


if __name__ == "__main__":
    dm = LokiDataModule(batch_size=256)
    lrvd = LokiTrainValDataset()
    num_classes = lrvd.n_classes
    label_encoder = lrvd.label_encoder
    logger = WandbLogger(project="loki")
    print(wandb.run.name)
    print(wandb.run.id)
    model = DtlModel(
        input_shape=(3, 300, 300),
        label_encoder=label_encoder,
        num_classes=num_classes,
        arch="resnet_dino450",
        transfer=True,
        num_train_layers=62,  # attention #resnet18 has here 62 layers, because a (FC) is counted as 2 y=xA^T+b
        wandb_name=wandb.run.name,
        learning_rate=0.0001,
    )
    trainer = pl.Trainer(
        precision=16,
        logger=logger,
        max_epochs=1,
        accelerator="mps",
        devices="auto",
        deterministic=True,
        # limit_train_batches=1
    )
    trainer.fit(model, dm)
    folder_path = f"loki/{wandb.run.id}/checkpoints/"
    file_pattern = folder_path + "*.ckpt"
    file_list = glob.glob(file_pattern)
    print(file_list[0])
    trainer.validate(model, dm, ckpt_path=file_list[0])
    trainer.test(model, dm, ckpt_path=file_list[0])
