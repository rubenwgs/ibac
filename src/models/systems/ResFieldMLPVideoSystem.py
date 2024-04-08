from typing import Any

from src.models.systems import MLPVideoSystem
import torch
from src.utils import compute_psnr
from torch.nn.functional import mse_loss
import numpy as np
import skvideo.io


class ResFieldMLPVideoSystem(MLPVideoSystem):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 validation_output_dir: str
                 ):
        super(ResFieldMLPVideoSystem, self).__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            validation_output_dir=validation_output_dir
        )

    def forward(self, coords, frame_ids):
        if not self.net.training:  # batchify coords to prevent OOM at inference time
            pred_list = []
            for _c, _f in zip(coords.split(1), frame_ids.split(1)):
                _pred = [self.net(__c, _f, input_time=__c[:, 0, 0]).detach().clone() for __c in
                         _c.split(100000, dim=1)]
                pred_list.append(torch.cat(_pred, dim=1))
            pred = torch.cat(pred_list, dim=0)
        else:
            pred = self.net(coords, frame_ids, input_time=coords[:, 0, 0])
        return pred

    def training_step(self, batch, batch_idx):
        pred = self(batch['coords'], batch['frame_ids'])
        loss = 1000. * mse_loss(pred, batch['data'])
        self.log('train/loss', loss, prog_bar=True, rank_zero_only=True, sync_dist=True)
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        metrics_dict = dict(metric_test_psnr=0, metric_train_psnr=0)
        if self.trainer.is_global_zero:
            test_pred = self(batch['coords'], batch['frame_ids'])
            test_psnr = compute_psnr(test_pred, batch['data']).mean()
            self.log('val/test_psnr', test_psnr, prog_bar=True, rank_zero_only=True, sync_dist=True)
            # calculate overfitting loss
            if 'train_coords' in batch:
                train_pred = self(batch['train_coords'], batch['train_frame_ids'])
                train_psnr = compute_psnr(train_pred, batch['train_data']).mean()
                self.log('val/train_psnr', train_psnr, prog_bar=True, rank_zero_only=True, sync_dist=True)

            metrics_dict = dict(metric_test_psnr=test_psnr.item(), metric_train_psnr=train_psnr.item())
        return metrics_dict

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        if self.trainer.is_global_zero:
            test_pred = self(batch['coords'], batch['frame_ids'])
            gt_video = self.dataset.data.view(*self.dataset.video_dims[:-1], test_pred.shape[-1])
            test_pred = test_pred.view(gt_video.shape)

            gt_video = (gt_video * 255).clip(0, 255).detach().cpu().numpy().astype(np.uint8)
            test_video = (test_pred * 255).clip(0, 255).detach().cpu().numpy().astype(np.uint8)

            skvideo.io.vwrite(self.get_save_path(f'video_rnd_it{self.global_step:06d}.mp4'), test_video)
            skvideo.io.vwrite(self.get_save_path(f'video_gt.mp4'), gt_video)
