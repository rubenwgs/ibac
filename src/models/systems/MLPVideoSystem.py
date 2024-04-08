from typing import Dict, Any, Optional
import os
import math
import skvideo.io
import numpy as np
from lightning import LightningModule
import torch.nn
from torch import Tensor
from torch.nn.functional import mse_loss
from src.utils import compute_psnr


class MLPVideoSystem(LightningModule):

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            validation_output_dir: str
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.validation_output_dir = validation_output_dir
        self.save_hyperparameters(logger=False, ignore=['net', 'validation_output_dir'])
        self.net = net
        self.net.setup()

    @property
    def save_dir(self):
        return self.validation_output_dir

    def get_save_path(self, filename):
        save_path = os.path.join(self.save_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path

    def preprocess_data(self, batch, stage):
        for key, val in batch.items():
            if torch.is_tensor(val):
                batch[key] = val.squeeze(0).to(self.device)
                if val.dtype == torch.float32 and self.trainer.precision == 16:
                    batch[key] = batch[key].to(torch.float16)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        print(optimizer)

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": 'step',
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step()

    def forward(self, x: Tensor) -> Tensor:
        if not self.net.training:  # batchify coords to prevent OOM at inference time

            pred_list = []

            for _c in x.split(1, dim=0):  # splits into a tuple of 300 tensors each of size [1, 26214, 3]
                _pred = [self.net(__c).detach().clone() for __c in _c.split(100000, dim=1)]  # each tensor
                pred_list.append(torch.cat(_pred, dim=1))
            pred = torch.cat(pred_list, dim=0)
            return pred
        else:
            return self.net(x)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        x = batch["coords"]
        y = batch["data"]
        x_hat = self(x)
        loss = 1000. * mse_loss(x_hat, y)
        self.log('train/loss', loss, prog_bar=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Any:
        x_hat_test_psnr = 0
        x_hat_train_psnr = 0
        if self.trainer.is_global_zero:
            x_test = batch["coords"]
            y = batch["data"]
            x_hat_test = self(x_test)
            x_hat_test_psnr = compute_psnr(x_hat_test, y).mean()
            self.log('val/test_psnr', x_hat_test_psnr, prog_bar=True, rank_zero_only=True, sync_dist=True)

            if 'train_coords' in batch.keys():
                x_train = batch['train_coords']
                x_hat_train = self(x_train)
                x_hat_train_psnr = compute_psnr(x_hat_train, batch["train_data"]).mean()
                self.log('val/train_psnr', x_hat_train_psnr, prog_bar=True, rank_zero_only=True, sync_dist=True)
        return dict(metric_test_psnr=x_hat_train_psnr, metric_train_psnr=x_hat_test_psnr)

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Any:
        if self.trainer.is_global_zero:
            x = batch["coords"]
            x_hat_test = self(x)

            gt_video = self.dataset.data.view(*self.dataset.video_dims[:-1], x_hat_test.shape[-1])
            test_pred = x_hat_test.view(gt_video.shape)

            gt_video = (gt_video * 255).clip(0, 255).detach().cpu().numpy().astype(np.uint8)
            test_video = (test_pred * 255).clip(0, 255).detach().cpu().numpy().astype(np.uint8)

            # save video
            skvideo.io.vwrite(self.get_save_path(f'video_rnd_it{self.global_step:06d}.mp4'), test_video)
            skvideo.io.vwrite(self.get_save_path(f'video_gt.mp4'), gt_video)

            # # log images
            # for i in range(test_pred.shape[0]):
            #     fname = 'img%06d.png' % i
            #     rnd_path = self.get_save_path(os.path.join(f'img_rnd_it{self.global_step:06d}', fname))
            #     gt_path = self.get_save_path(os.path.join(f'img_gt', fname))
            #
            #     cv2.imwrite(rnd_path, test_video[i, :, ::-1])
            #     cv2.imwrite(gt_path, gt_video[i, :, ::-1])

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.dataset = self.trainer.datamodule.train_dataloader().dataset
        self.preprocess_data(batch, 'train')

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.dataset = self.trainer.datamodule.val_dataloader().dataset
        self.preprocess_data(batch, 'validation')

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.dataset = self.trainer.datamodule.test_dataloader().dataset
        self.preprocess_data(batch, 'test')

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.dataset = self.trainer.datamodule.predict_dataloader().dataset
        self.preprocess_data(batch, 'predict')
