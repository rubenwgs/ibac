from typing import Tuple

import numpy as np
import torch

from src.utils.video_utils import create_random_video_train_test_data


class BaseVideoDataset:
    path_to_video: str
    unseen_strategy: str
    test_fraction: float
    samples: int
    seed: int
    split: str

    def setup(
            self,
            path_to_video: str,
            unseen_strategy: str = 'random',
            test_fraction: float = 0.1,
            samples: int = 200000,
            split: str = 'train',
            seed: int = 42
    ):

        self.path_to_video = path_to_video
        self.unseen_strategy = unseen_strategy
        self.test_fraction = test_fraction
        self.samples = samples
        self.seed = seed
        self.split = split

        self.__setup_strategy()

    def __setup_strategy(self):
        match self.unseen_strategy:
            case 'random':
                temp_data = create_random_video_train_test_data(self.path_to_video, self.test_fraction, 3, self.seed)
            case 'unseen_time':
                raise NotImplementedError
            case _:
                raise ValueError

        self.video_data_tensor = temp_data[0]
        self.video_coordinate_tensor = temp_data[1]
        self.test_data = temp_data[2]
        self.test_coordinates = temp_data[3]
        self.test_frames = temp_data[4]  # T
        self.train_data = temp_data[5]  # T x (PPF - Ns) x C
        self.train_coordinates = temp_data[6]  # T x (PPF - Ns) x C
        self.train_frames = temp_data[7]  # T
        self.frames = self.video_data_tensor.shape[0]
        self.video_dims = temp_data[8]
        self.__set_data()

    def __set_data(self):

        if self.split == 'train':

            self.samples = self.samples // self.frames
            self.data = self.train_data
            self.coordinates = self.train_coordinates
            self.frames_idx_list = self.train_frames

        elif self.split == 'test':

            self.data = self.test_data
            self.coordinates = self.test_coordinates
            self.frames_idx_list = self.test_frames

        elif self.split == 'predict':
            self.data = self.video_data_tensor
            self.coordinates = self.video_coordinate_tensor
            self.frames_idx_list = torch.arange(self.frames).long()

    def sample_data(self):
        d = self.data
        c = self.coordinates
        batch = dict()

        if self.split in ['train']:
            t = torch.arange(self.frames, device=self.coordinates.device).unsqueeze(-1).repeat(1,
                                                                                               self.samples).view(
                -1)
            y = torch.randint(0, c.shape[1], size=(t.shape[0],), device=self.coordinates.device)
            d = self.data[t, y]
            c = self.coordinates[t, y]
        else:
            # add training coordinates to the batch to log the overfitting loss
            batch.update({
                'train_coords': self.train_coordinates.view(self.train_coordinates.shape[0], -1,
                                                            self.train_coordinates.shape[-1]).float(),
                'train_data': self.train_data.view(self.train_data.shape[0], -1,
                                                   self.train_data.shape[-1]).float(),
                'train_frame_ids': self.train_frames,
            })
        batch.update({
            'coords': c.view(self.frames_idx_list.shape[0], -1, c.shape[-1]).float(),  # T,S,3
            'data': d.view(self.frames_idx_list.shape[0], -1, d.shape[-1]).float(),  # T,S,3
            'frame_ids': self.frames_idx_list,  # T
        })
        return batch
