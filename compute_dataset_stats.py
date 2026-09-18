"""Precompute Inception `pool_3` statistics for the CIFAR-10 training set.

`evaluation.load_dataset_stats` expects a `assets/stats/cifar10_stats.npz`
file containing reference activations for FID/KID computation. Since this
codebase now uses a PyTorch Inception-v3 network (see `evaluation.py`)
instead of the original TensorFlow/TF-Hub one, any previously downloaded
stats file is incompatible and must be recomputed with this script:

  python compute_dataset_stats.py
"""
import os

import numpy as np
import torchvision

import evaluation


def main():
  os.makedirs('assets/stats', exist_ok=True)
  dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
  images = np.stack([np.array(img) for img, _ in dataset], axis=0)  # (N, 32, 32, 3) uint8

  inception_model = evaluation.get_inception_model()
  pools = []
  batch_size = 250
  for i in range(0, len(images), batch_size):
    batch = images[i:i + batch_size]
    latents = evaluation.run_inception_distributed(batch, inception_model)
    pools.append(latents['pool_3'])
    print(f'Processed {i + len(batch)}/{len(images)} images')

  pool_3 = np.concatenate(pools, axis=0)
  np.savez_compressed('assets/stats/cifar10_stats.npz', pool_3=pool_3)
  print('Saved assets/stats/cifar10_stats.npz')


if __name__ == '__main__':
  main()
