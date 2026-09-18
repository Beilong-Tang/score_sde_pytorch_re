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

"""Utility functions for computing FID/Inception Score/KID with a PyTorch Inception-v3 model."""

import numpy as np
import scipy.linalg
import scipy.special
import torch
import torch.nn.functional as F
import torchvision

INCEPTION_DEFAULT_IMAGE_SIZE = 299


def get_inception_model(device=None):
  """Load a pretrained Inception-v3 network used for FID/IS/KID evaluation."""
  device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = torchvision.models.inception_v3(
    weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
  model.eval()
  model.to(device)

  # `pool_3` are the 2048-d activations right before the final FC layer,
  # the standard features used for FID/KID.
  pool_features = {}

  def _hook(module, inputs, output):
    pool_features['pool_3'] = torch.flatten(output, 1)

  model.avgpool.register_forward_hook(_hook)
  model._pool_features = pool_features
  return model


def load_dataset_stats(config):
  """Load the pre-computed dataset statistics."""
  if config.data.dataset == 'CIFAR10':
    filename = 'assets/stats/cifar10_stats.npz'
  else:
    raise ValueError(f'Dataset {config.data.dataset} stats not found.')

  with open(filename, 'rb') as fin:
    stats = np.load(fin)
    return {'pool_3': stats['pool_3']}


@torch.no_grad()
def run_inception_distributed(inputs, inception_model, num_batches=1):
  """Run the Inception network on a batch of images.

  Args:
    inputs: A numpy array of images with shape (N, H, W, C), uint8 in [0, 255].
    inception_model: The model returned by `get_inception_model`.
    num_batches: Split `inputs` into this many chunks to bound memory use.

  Returns:
    A dict with keys `pool_3` (2048-d activations) and `logits` (1000-d class
      logits).
  """
  device = next(inception_model.parameters()).device
  images = torch.from_numpy(inputs).to(device).float() / 255.
  images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
  images = F.interpolate(
    images, size=(INCEPTION_DEFAULT_IMAGE_SIZE, INCEPTION_DEFAULT_IMAGE_SIZE),
    mode='bilinear', align_corners=False)
  # Rescale to [-1, 1], the range `torchvision`'s pretrained Inception-v3
  # expects when `transform_input=True`.
  images = (images - 0.5) * 2.

  pools, logits = [], []
  for chunk in torch.chunk(images, num_batches):
    chunk_logits = inception_model(chunk)
    pools.append(inception_model._pool_features['pool_3'].cpu().numpy())
    logits.append(chunk_logits.cpu().numpy())

  return {
    'pool_3': np.concatenate(pools, axis=0),
    'logits': np.concatenate(logits, axis=0),
  }


def inception_score_from_logits(logits, num_splits=10):
  """Compute the Inception Score from classifier logits."""
  logits = np.asarray(logits, dtype=np.float64)
  probs = scipy.special.softmax(logits, axis=1)
  n = probs.shape[0]
  scores = []
  for i in range(num_splits):
    part = probs[i * n // num_splits:(i + 1) * n // num_splits]
    marginal = np.mean(part, axis=0, keepdims=True)
    kl = part * (np.log(part + 1e-12) - np.log(marginal + 1e-12))
    scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
  return float(np.mean(scores))


def frechet_distance_from_activations(real_activations, fake_activations):
  """Compute the Frechet distance (FID) between two sets of activations."""
  real_activations = np.asarray(real_activations, dtype=np.float64)
  fake_activations = np.asarray(fake_activations, dtype=np.float64)

  mu1, mu2 = real_activations.mean(axis=0), fake_activations.mean(axis=0)
  sigma1 = np.cov(real_activations, rowvar=False)
  sigma2 = np.cov(fake_activations, rowvar=False)

  diff = mu1 - mu2
  covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if np.iscomplexobj(covmean):
    covmean = covmean.real

  return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2. * covmean))


def kernel_distance_from_activations(real_activations, fake_activations,
                                     num_subsets=100, max_subset_size=1000):
  """Estimate the Kernel Inception Distance (KID) with a cubic polynomial kernel."""
  real_activations = np.asarray(real_activations, dtype=np.float64)
  fake_activations = np.asarray(fake_activations, dtype=np.float64)
  n_real, n_fake = real_activations.shape[0], fake_activations.shape[0]
  m = min(min(n_real, n_fake), max_subset_size)
  d = real_activations.shape[1]

  def poly_kernel(x, y):
    return (x.dot(y.T) / d + 1.) ** 3

  rng = np.random.RandomState(0)
  total = 0.
  for _ in range(num_subsets):
    x = real_activations[rng.choice(n_real, m, replace=False)]
    y = fake_activations[rng.choice(n_fake, m, replace=False)]

    kxx = poly_kernel(x, x)
    kyy = poly_kernel(y, y)
    kxy = poly_kernel(x, y)

    total += ((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
              + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
              - 2 * kxy.mean())

  return float(total / num_subsets)
