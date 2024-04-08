from typing import override
import torch.utils.data
from src.data.datasets.BaseVideoDataset import BaseVideoDataset

# Torch Wrapper
class TorchVideoDataset(torch.utils.data.Dataset, BaseVideoDataset):

    def __init__(self,
                 path_to_video: str,
                 unseen_strategy: str = 'random',
                 test_fraction: float = 0.1,
                 samples: int = 200000,
                 split: str = 'train',
                 seed: int = 42
                 ) -> None:
        self.setup(path_to_video=path_to_video,
                   unseen_strategy=unseen_strategy,
                   test_fraction=test_fraction,
                   samples=samples,
                   split=split,
                   seed=seed)

    def __len__(self):
        return 1

    @override
    def __getitem__(self, index):
        batch = self.sample_data()
        batch['index'] = index
        return batch
