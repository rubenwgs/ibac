from typing import Tuple
from torch import Tensor

import torch
import numpy as np

np.float = float
np.int = int
import skvideo.io
import skvideo.datasets


def load_video_data(path_to_video: str) -> np.ndarray:
    if 'skvideo.datasets' in path_to_video:
        path_to_video = eval(path_to_video)()

    if 'npy' in path_to_video:
        video = np.load(path_to_video)
        return video  # T,H,W,C
    elif 'mp4' in path_to_video:
        video = skvideo.io.vread(path_to_video).astype(np.float64) / 255.

        return video  # T,H,W,C
    else:
        raise FileNotFoundError


def get_metadata(path_to_video: str) -> dict:
    frames = load_video_data(path_to_video).shape[0]
    return {'frames': frames}


def generate_random_mask(
        video_dims: Tuple[int, ...],
        test_fraction: float = 0.1,
        seed: int = 42
):
    frames = video_dims[0]
    pixels_per_frame = np.prod(video_dims[1:-1])
    rng = np.random.default_rng(seed)

    samples_per_frame = int(test_fraction * pixels_per_frame)

    test_samples_idx_list = [rng.choice(pixels_per_frame, size=samples_per_frame, replace=False) for _ in
                             range(frames)]  # T x Ns
    train_samples_idx_list = [np.delete(np.arange(pixels_per_frame), test_samples_idx_list[i]) for i in
                              range(frames)]  # T x (PPF - Ns)

    test_idx_list = np.stack(test_samples_idx_list).reshape(-1)  # (T * Ns)
    train_idx_list = np.stack(train_samples_idx_list).reshape(-1)  # (T * (PPF - Ns))

    return test_idx_list, train_idx_list


def create_random_video_train_test_data(
        path_to_video: str,
        test_fraction: float = 0.1,
        mesh_dim: int = 3,
        seed: int = 42
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tuple[int, ...]]:
    video = load_video_data(path_to_video)
    video_dims = video.shape

    frames = video_dims[0]
    pixels_per_frame = np.prod(video_dims[1:-1])
    color_channels = video_dims[-1]

    video_data_tensor = torch.from_numpy(video).float().view(frames, pixels_per_frame,
                                                             color_channels)  # T x PPF x C

    video_coordinate_tensor = create_video_mesh_grid(video_dims, dim=mesh_dim)

    samples_per_frame = int(test_fraction * pixels_per_frame)

    test_idx_list, train_idx_list = generate_random_mask(video_dims, test_fraction, seed)

    frame_idx_list = np.arange(0, frames)[:, None]  # T x 1

    test_frames_idx_list = frame_idx_list.repeat(samples_per_frame, axis=1).flatten()  # (T * Ns)
    train_frames_idx_list = frame_idx_list.repeat((pixels_per_frame - samples_per_frame),
                                                  axis=1).flatten()  # T x (PPF - Ns)

    test_coordinates = video_coordinate_tensor[test_frames_idx_list, test_idx_list].reshape(frames, -1,
                                                                                            color_channels)  # T x Ns x C
    train_coordinates = video_coordinate_tensor[train_frames_idx_list, train_idx_list].reshape(frames, -1,
                                                                                               color_channels)  # T x (PPF - Ns) x C

    test_data = video_data_tensor[test_frames_idx_list, test_idx_list].reshape(frames, -1,
                                                                               color_channels)  # T x Ns x C
    train_data = video_data_tensor[train_frames_idx_list, train_idx_list].reshape(frames, -1,
                                                                                  color_channels)  # T x (PPF - Ns) x C

    train_frames = torch.arange(frames)  # T
    test_frames = torch.arange(frames)  # T

    # set the class parameters

    test_data = test_data.float()  # T x Ns x C
    test_coordinates = test_coordinates.float()  # T x Ns x C
    test_frames = test_frames.long()  # T

    train_data = train_data.float()  # T x (PPF - Ns) x C
    train_coordinates = train_coordinates.float()  # T x (PPF - Ns) x C
    train_frames = train_frames.long()  # T

    return (video_data_tensor,
            video_coordinate_tensor,
            test_data,
            test_coordinates,
            test_frames,
            train_data,
            train_coordinates,
            train_frames,
            video_dims)


def create_video_mesh_grid(
        video_dims: Tuple[int, ...],
        dim: int = 3
) -> Tensor:
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""

    domain_axis = [np.linspace(-1, 1, video_dims[i]) for i in range(dim)]
    domain_mesh_grid = np.meshgrid(*domain_axis, indexing='ij')
    raveled_mesh_grid = [x.ravel() for x in domain_mesh_grid]
    stacked_mesh_grid = np.dstack(raveled_mesh_grid).astype(np.float32)
    flattened_mesh_grid_tensor = torch.tensor(stacked_mesh_grid).view(-1, dim)

    return flattened_mesh_grid_tensor.view(video_dims[0], -1, video_dims[-1])
