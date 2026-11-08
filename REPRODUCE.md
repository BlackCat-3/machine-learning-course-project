# 运行说明

## 环境与数据

原实验主要在 Windows、NVIDIA RTX 4060 Laptop GPU、PyTorch 2.11.0+cu128 上运行。各次运行的真实配置和环境以 `experiment_records` 下记录为准。不同硬件重新训练可能产生数值差异，本整理过程没有重训模型。

任务二、三可参考 `task2_emotions/requirements.txt`，任务四参考 `task4_vessels/requirements.txt`。任务一另需 scikit-learn、threadpoolctl。建议各任务使用独立虚拟环境；安装 PyTorch 时按目标机器选择匹配的 CUDA 版本。部分原脚本默认数据位于 `D:/Downloads`，可通过 `--zip` 指定新位置；少数早期任务一脚本以顶部 `ZIP` 常量指定压缩包，请在新副本中按本机位置修改。

## 任务一

工作目录为 `task1_plants`。正式入口放在 `outputs/` 是原实验布局，保留以兼容脚本之间的相对引用。

```powershell
python outputs/plant_frozen_features.py --zip "你的路径/2026task1.zip" --backbone dinov2_vitb14 --size 336 --foreground --views 2
python outputs/plant_feature_classifier.py --features work/features_dinov2_vitb14_336_fg_v2 --out outputs/dinov2_base336_augmented --stage cv --training-views augmented
python outputs/plant_feature_classifier.py --features work/features_dinov2_vitb14_336_fg_v2 --out outputs/dinov2_base336_augmented --stage predict --training-views augmented
python outputs/plant_resample_search.py
python outputs/plant_seed_confirm.py --search outputs/dinov2_resample_search --candidate resample_1001
```

最终历史选择为种子 1001；从头重新运行搜索可能受环境和优化数值差异影响，不保证仍在同一种子首次超过基准。`plant_5fold.py` 对应早期 ResNet18 对照，`plant_variant_experiments.py`、`plant_tta.py` 等对应过程实验。DINOv2 权重由原代码通过官方 torch.hub 获取。

## 任务二

工作目录为 `task2_emotions`。先准备数据，提交模板用课程提供的任务二模板：

```powershell
python prepare_data.py --zip "你的路径/2026task2.zip" --submission "你的路径/submission.csv"
python train.py --output-dir outputs/resnet18_main --epochs 18 --seed 42
python train.py --output-dir outputs/resnet18_mixup_s43 --epochs 18 --seed 43 --mixup-alpha 0.2 --mixup-prob 1 --dropout 0.3 --weight-decay 0.001
```

两次基础训练后，需要生成验证概率、早期全量双模型及 `outputs/submission.csv`，再执行 `improve.py --stage all` 和 `seed_erasing_study.py --stage all`。这些脚本会检查既有基线文件，不应跳过依赖步骤。完整顺序与命令见 `HISTORY.md` 及 `train_final.ps1`；PowerShell 脚本要求该任务目录内已有 `.venv`。

最终方案为四个全量 ResNet 模型、水平翻转 TTA 和等权概率平均。对应核验记录：`experiment_records/task2/outputs/seed_erasing_20261016/`。后续未完成的骨干实验保留实际状态，不作为最终提交。

## 任务三

在仓库根目录运行。课程压缩包解压后应形成 `task3_fovea/data/detection/`，包含 train、test、train_location、坐标 CSV 和提交模板。ResNet18、ResNet34 预训练权重分别放在仓库根目录 `.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` 与 `resnet34-b627a593.pth`，来源为 torchvision 官方权重。

```powershell
python task3_fovea/study.py --stage all
python task3_fovea/robust_heatmap.py --stage all
```

第一条建立初版分组四折与全量模型，第二条执行最终 ResNet34 双种子方案。完整缓存和权重恢复后可用 `--stage final` 重新导出；仅有代码与历史 JSON 不能跳过训练。后续精修未通过门槛，入口及实际结果见 `HISTORY.md` 和历史记录。

## 任务四

在仓库根目录运行：

```powershell
python task4_vessels/prepare.py --zip "你的路径/2026task-4.zip"
python task4_vessels/pipeline.py --run baseline_v1
python task4_vessels/fr_unet_transfer.py
```

后两条分别执行从零训练基线及 CHASE 微调。CHASE 与最终 DRIVE 权重需从 FR-UNet 固定版本获取，放到 `task4_vessels/third_party/fr_unet/pretrained_weights/CHASEDB1/checkpoint-epoch40.pth`、`DRIVE/checkpoint-epoch40.pth`；来源和 SHA256 见该目录的 provenance 文件。

最终 DRIVE 推理不需要重新微调，但 `drive_pretrained_infer.py` 会校验来源核对记录。将历史 `experiment_records/task4/outputs/improve_v2_20261101/source_identity/` 的两个 `training_identity.json`、`training_label_identity.json` 恢复到同名运行目录，再执行：

```powershell
python task4_vessels/drive_pretrained_infer.py --tta 8 --refresh
```

输出候选位于 `task4_vessels/outputs/improve_v2_20261101/official_drive_d4/hysteresis_official/submission.csv`。`verify_v2.py` 还要求上一版 CHASE 提交及其原始核对记录存在；未恢复这些材料时不能直接声称完整历史核验通过。报告对应的最终 CSV 已原样收录在 `results/task4_submission.csv`。

## 检查范围

整理包已检查源码语法、复制哈希、第三方许可证、最终提交哈希和文件体积。没有在一个全新环境重跑完整训练，也没有用整理包重新预测全部任务。历史文件中的原电脑绝对路径用于定位实验来源，迁移机器时需要按实际目录恢复。课程原数据及完整权重仍由作者本地保留用于复查。
