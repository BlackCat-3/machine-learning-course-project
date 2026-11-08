# Upstream source and local compatibility change

Official repository: https://github.com/lseventeen/FR-UNet

Pinned revision: `1856bc206055fdfccde184d0ebf9ac8c3207387d`.

The upstream MIT license is retained in `LICENSE`. `provenance.json` records the original downloaded file SHA256 values, including the original CHASEDB1 checkpoint. The original FR-UNet architecture file is unchanged.

One compatibility change was made in `models/utils.py`: import `trunc_normal_` from `torch.nn.init` instead of `timm.models.layers`. FR-UNet has no Linear layers using that initialization branch; no trained parameters were changed by this edit. The run's `source_manifest.json` records the adapted Python source hashes.

The original checkpoint is from the upstream `pretrained_weights/CHASEDB1/checkpoint-epoch40.pth` path, and its embedded configuration names CHASEDB1. The loader uses PyTorch's restricted weights-only loading with the legacy `bunch.Bunch` metadata mapped to a standard OrderedDict; it does not execute a downloaded pickle class. DataParallel prefixes are removed, then all parameters are loaded strictly.

The surrounding training, data handling, threshold selection, CSV conversion and auditing code in `../../fr_unet_transfer.py` is task-specific. It uses the existing image-level development split. Local fine-tuning differs from the paper's original training procedure and is not claimed as a paper reproduction.
