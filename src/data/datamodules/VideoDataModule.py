from typing import Any, Optional
from typing import override

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.data.wrappers import TorchIterableVideoDataset, TorchVideoDataset


class VideoDataModule(LightningDataModule):
    path_to_video: str
    test_fraction: float
    unseen_strategy: str
    samples: int
    seed: int
    num_workers: int
    batch_size: int
    pin_memory: bool
    persistent_workers: bool

    def __init__(self,
                 path_to_video: str,
                 unseen_strategy: str = 'random',
                 test_fraction: float = 0.1,
                 batch_size: int = 1,
                 samples: int = 200000,
                 num_workers: int = 5,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 seed: int = 42
                 ) -> None:
        super().__init__()

        self.path_to_video = path_to_video
        self.test_fraction = test_fraction
        self.seed = seed
        self.unseen_strategy = unseen_strategy
        self.samples = samples

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.train_dataset: Optional[TorchIterableVideoDataset] = None
        self.predict_dataset: Optional[TorchVideoDataset] = None
        self.test_dataset: Optional[TorchVideoDataset] = None
        self.validation_dataset: Optional[TorchVideoDataset] = None

    @override
    def setup(self, stage: Optional[str] = None):

        if stage in [None, 'fit']:
            self.train_dataset = TorchIterableVideoDataset(path_to_video=self.path_to_video,
                                                           unseen_strategy=self.unseen_strategy,
                                                           test_fraction=self.test_fraction,
                                                           samples=self.samples,
                                                           seed=self.seed,
                                                           split='train')
        if stage in [None, 'fit', 'validate']:
            self.validation_dataset = TorchVideoDataset(path_to_video=self.path_to_video,
                                                        unseen_strategy=self.unseen_strategy,
                                                        test_fraction=self.test_fraction,
                                                        samples=self.samples,
                                                        seed=self.seed,
                                                        split='test')
        if stage in [None, 'test']:
            self.test_dataset = TorchVideoDataset(path_to_video=self.path_to_video,
                                                  unseen_strategy=self.unseen_strategy,
                                                  test_fraction=self.test_fraction,
                                                  samples=self.samples,
                                                  seed=self.seed,
                                                  split='test')
        if stage in [None, 'predict']:
            self.predict_dataset = TorchVideoDataset(path_to_video=self.path_to_video,
                                                     unseen_strategy=self.unseen_strategy,
                                                     test_fraction=self.test_fraction,
                                                     samples=self.samples,
                                                     seed=self.seed,
                                                     split='predict')

    @override
    def prepare_data(self) -> None:
        pass

    def general_loader(self, dataset):
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            sampler=None,
            persistent_workers=self.persistent_workers
        )

    @override
    def train_dataloader(self) -> DataLoader[Any]:
        return self.general_loader(self.train_dataset)

    @override
    def val_dataloader(self) -> DataLoader[Any]:
        return self.general_loader(self.validation_dataset)

    @override
    def test_dataloader(self) -> DataLoader[Any]:
        return self.general_loader(self.test_dataset)

    @override
    def predict_dataloader(self) -> DataLoader[Any]:
        return self.general_loader(self.predict_dataset)

