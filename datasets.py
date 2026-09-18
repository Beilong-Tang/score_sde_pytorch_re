# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test data loaders from config files.

This is a pure PyTorch data pipeline (no TensorFlow/TFDS dependency) and
currently only supports the CIFAR-10 dataset.
"""
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


class _RepeatingLoader:
  """Iterates a `DataLoader` for `num_epochs` epochs, or forever if `None`."""

  def __init__(self, dataloader, num_epochs=None, sampler=None):
    self.dataloader = dataloader
    self.num_epochs = num_epochs
    self.sampler = sampler

  def __iter__(self):
    epoch = 0
    while self.num_epochs is None or epoch < self.num_epochs:
      if isinstance(self.sampler, DistributedSampler):
        self.sampler.set_epoch(epoch)
      for batch in self.dataloader:
        yield batch
      epoch += 1

  def __len__(self):
    return len(self.dataloader)


def _make_transform(config, uniform_dequantization, evaluation):
  """Build the per-image preprocessing function used by the data loaders."""
  image_size = config.data.image_size
  random_flip = config.data.random_flip and not evaluation

  def transform(pil_img):
    img = TF.to_tensor(pil_img)  # CHW, float32 in [0, 1]
    if img.shape[-2] != image_size or img.shape[-1] != image_size:
      img = TF.resize(img, [image_size, image_size], antialias=True)
    if random_flip and torch.rand(()) < 0.5:
      img = torch.flip(img, dims=[-1])
    if uniform_dequantization:
      img = (torch.rand_like(img) + img * 255.) / 256.
    return img

  return transform


def _collate_fn(batch):
  images = torch.stack([img for img, _ in batch], dim=0)
  labels = torch.tensor([label for _, label in batch], dtype=torch.long)
  return dict(image=images, label=labels)


def get_dataset(config, uniform_dequantization=False, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  if config.data.dataset != 'CIFAR10':
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not supported. This PyTorch-only '
      'data pipeline currently supports CIFAR10 only.')

  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  num_workers = getattr(config.data, 'num_workers', 4)
  data_root = getattr(config.data, 'root', './data')
  num_epochs = None if not evaluation else 1

  distributed = dist.is_available() and dist.is_initialized()
  # Only rank 0 downloads to avoid every process racing to write the same files.
  if not distributed or dist.get_rank() == 0:
    torchvision.datasets.CIFAR10(root=data_root, train=True, download=False)
    torchvision.datasets.CIFAR10(root=data_root, train=False, download=False)
  if distributed:
    dist.barrier()

  transform = _make_transform(config, uniform_dequantization, evaluation)
  train_set = torchvision.datasets.CIFAR10(
    root=data_root, train=True, download=False, transform=transform)
  eval_set = torchvision.datasets.CIFAR10(
    root=data_root, train=False, download=False, transform=transform)

  def make_loader(dataset):
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    dataloader = DataLoader(
      dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
      drop_last=True, num_workers=num_workers, pin_memory=True, collate_fn=_collate_fn,
      persistent_workers=num_workers > 0,
      prefetch_factor=4 if num_workers > 0 else None)
    return _RepeatingLoader(dataloader, num_epochs=num_epochs, sampler=sampler)

  train_ds = make_loader(train_set)
  eval_ds = make_loader(eval_set)
  return train_ds, eval_ds, None
