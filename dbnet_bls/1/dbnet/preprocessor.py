# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from typing import Any, List, Tuple, Union, Callable, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.functional import pad
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T



class Resize(T.Resize):
    def __init__(
        self,
        size: Tuple[int, int],
        interpolation=F.InterpolationMode.BILINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        super().__init__(size, interpolation)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[-2] / img.shape[-1]
        if not self.preserve_aspect_ratio or (target_ratio == actual_ratio):
            return super().forward(img)
        else:
            # Resize
            if actual_ratio > target_ratio:
                tmp_size = (self.size[0], max(int(self.size[0] / actual_ratio), 1))
            else:
                tmp_size = (max(int(self.size[1] * actual_ratio), 1), self.size[1])

            # Scale image
            img = F.resize(img, tmp_size, self.interpolation)
            # Pad (inverted in pytorch)
            _pad = (0, self.size[1] - img.shape[-1], 0, self.size[0] - img.shape[-2])
            if self.symmetric_pad:
                half_pad = (math.ceil(_pad[1] / 2), math.ceil(_pad[3] / 2))
                _pad = (half_pad[0], _pad[1] - half_pad[0], half_pad[1], _pad[3] - half_pad[1])
            return pad(img, _pad)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        _repr = f"output_size={self.size}, interpolation='{interpolate_str}'"
        if self.preserve_aspect_ratio:
            _repr += f", preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}"
        return f"{self.__class__.__name__}({_repr})"



def _multithread_exec(func: Callable[[Any], Any], seq: Iterable[Any], threads: Optional[int] = None) -> Iterable[Any]:
    """Execute a given function in parallel for each element of a given sequence

    Example::
        >>> from doctr.utils.multithreading import multithread_exec
        >>> entries = [1, 4, 8]
        >>> results = multithread_exec(lambda x: x ** 2, entries)

    Args:
        func: function to be executed on each element of the iterable
        seq: iterable
        threads: number of workers to be used for multiprocessing

    Returns:
        iterable of the function's results using the iterable as inputs
    """

    threads = threads if isinstance(threads, int) else min(16, mp.cpu_count())
    # Single-thread
    if threads < 2:
        results = map(func, seq)
    # Multi-threading
    else:
        with ThreadPool(threads) as tp:
            results = tp.map(func, seq)  # type: ignore[assignment]
    return results



class PreProcessor(nn.Module):
    """Implements an abstract preprocessor object which performs casting, resizing, batching and normalization.

    Args:
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
    """

    def __init__(
        self,
        output_size: Tuple[int, int],
        batch_size: int,
        mean: Tuple[float, float, float] = (.5, .5, .5),
        std: Tuple[float, float, float] = (1., 1., 1.),
        fp16: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.resize: T.Resize = Resize(output_size, **kwargs)
        # Perform the division by 255 at the same time
        self.normalize = T.Normalize(mean, std)

    def batch_inputs(
        self,
        samples: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Gather samples into batches for inference purposes

        Args:
            samples: list of samples of shape (C, H, W)

        Returns:
            list of batched samples (*, C, H, W)
        """

        num_batches = int(math.ceil(len(samples) / self.batch_size))
        batches = [
            torch.stack(samples[idx * self.batch_size: min((idx + 1) * self.batch_size, len(samples))], dim=0)
            for idx in range(int(num_batches))
        ]

        return batches

    def sample_transforms(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if x.ndim != 3:
            raise AssertionError("expected list of 3D Tensors")
        if isinstance(x, np.ndarray):
            if x.dtype not in (np.uint8, np.float32):
                raise TypeError("unsupported data type for numpy.ndarray")
            x = torch.from_numpy(x.copy()).permute(2, 0, 1)
        elif x.dtype not in (torch.uint8, torch.float16, torch.float32):
            raise TypeError("unsupported data type for torch.Tensor")
        # Resizing
        x = self.resize(x)
        # Data type
        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32).div(255).clip(0, 1)
        x = x.to(dtype=torch.float32)

        return x

    def __call__(
        self,
        x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]]
    ) -> List[torch.Tensor]:
        """Prepare document data for model forwarding

        Args:
            x: list of images (np.array) or tensors (already resized and batched)
        Returns:
            list of page batches
        """

        # Input type check
        if isinstance(x, (np.ndarray, torch.Tensor)):
            if x.ndim != 4:
                raise AssertionError("expected 4D Tensor")
            if isinstance(x, np.ndarray):
                if x.dtype not in (np.uint8, np.float32):
                    raise TypeError("unsupported data type for numpy.ndarray")
                x = torch.from_numpy(x.copy()).permute(0, 3, 1, 2)
            elif x.dtype not in (torch.uint8, torch.float16, torch.float32):
                raise TypeError("unsupported data type for torch.Tensor")
            # Resizing
            if x.shape[-2] != self.resize.size[0] or x.shape[-1] != self.resize.size[1]:
                x = F.resize(x, self.resize.size, interpolation=self.resize.interpolation)
            # Data type
            if x.dtype == torch.uint8:
                x = x.to(dtype=torch.float32).div(255).clip(0, 1)
            x = x.to(dtype=torch.float32)
            batches = [x]

        elif isinstance(x, list) and all(isinstance(sample, (np.ndarray, torch.Tensor)) for sample in x):
            # Sample transform (to tensor, resize)
            samples = _multithread_exec(self.sample_transforms, x)
            # Batching
            batches = self.batch_inputs(samples)  # type: ignore[arg-type]
        else:
            raise TypeError(f"invalid input type: {type(x)}")

        # Batch transforms (normalize)
        batches = _multithread_exec(self.normalize, batches)  # type: ignore[assignment]

        return batches
