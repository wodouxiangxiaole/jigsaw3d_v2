import os
import torch
import hydra
import lightning.pytorch as pl
from jigsaw_latent.data.data_module import DataModule
from lightning.pytorch.callbacks import LearningRateMonitor


def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.checkpoint_monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, lr_monitor]


@hydra.main(version_base=None, config_path="config_latent", config_name="global_config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.train_seed, workers=True)

    # create directories for training outputs
    os.makedirs(os.path.join(cfg.experiment_output_path, "training"), exist_ok=True)

    # initialize data
    data_module = DataModule(cfg)

    # initialize model
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)

    # load pretrained weights for pn++
    if cfg.model.encoder_weights_path is not None:
        encoder_weights = torch.load(cfg.model.encoder_weights_path)['state_dict']
        model.encoder.load_state_dict({k.replace('module.', ''): v for k, v in encoder_weights.items()})

    # initialize logger
    logger = hydra.utils.instantiate(cfg.logger)

    # initialize callbacks
    callbacks = init_callbacks(cfg)

    # initialize trainer
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.trainer)

    # check the checkpoint
    if cfg.ckpt_path is not None:
        assert os.path.exists(cfg.ckpt_path), "Error: Checkpoint path does not exist."
    
    # start training
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    main()