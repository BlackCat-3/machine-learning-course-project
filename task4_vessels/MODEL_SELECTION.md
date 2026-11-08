# 公开模型选型（2026-11-01）

用户要求优先使用网上强模型。本次实际检索了作者论文与官方代码，不能据此宣称存在适用于所有数据的唯一“最强模型”。候选如下。

## R2-V2（冠军级候选，未用于本地预测）

- [作者官方仓库](https://github.com/j-morano/R2-V2)称其为 MICCAI 2025 GAVE 挑战赛获胜方法。
- [作者发布的 bv 权重](https://huggingface.co/j-morano/R2-V2)面向血管分割；av 版本更偏向动静脉区分。代码、配置、权重可公开获取。
- [作者技术报告源码](https://github.com/j-morano/R2-V2_materials/blob/main/Report/main.tex)写明训练数据包括 RITE 等数据，并称最终模型使用全部 386 张图。
- [RITE 原始发布页](https://eye.medicine.uiowa.edu/rite-dataset)明确 RITE 继承 DRIVE 的 40 张眼底照片。本作业的 20/20 数量和 565×584 尺寸与 DRIVE 一致，但尚未逐图与原版 DRIVE 匹配。因而不能把 R2-V2 官方权重对本作业的分数作为已确认无训练重叠的泛化结果。本次没有下载其预测或测试标签，也未使用该权重。

## FR-UNet（本次实际接入的强基线）

- [官方仓库](https://github.com/lseventeen/FR-UNet)：IEEE JBHI 2022，Full-Resolution Network and Dual-Threshold Iteration for Retinal Vessel and Coronary Angiograph Segmentation；MIT 许可证。
- 作者公开完整训练/测试代码和不同数据集的权重，适合保存来源并复现课程提交。
- 实际下载 CHASEDB1 权重，不下载 DRIVE 权重。其内嵌训练配置同样标识 CHASEDB1，参数量为 7,374,347。
- 保留原始全分辨率多尺度网络，灰度输入及逐图 min-max 归一化。适配现有 PyTorch 后，先在既有 4 张开发验证图上测试，再用另外 16 张微调，最后按选定更新次数用全部 20 张微调。
- 本轮设置：64×64 图块、batch 16、1,200 次更新、每 200 步验证、AdamW、初始学习率 1e-4 余弦降至 2e-5，BCE + 0.5 Dice 损失；随机旋转、翻转与亮度/gamma 增强。验证阈值网格及四翻转 TTA 沿用第一版。
- 为明确比较来源，本轮没有启用论文的双阈值迭代后处理；效果来自预训练、结构、预处理及本地微调的组合，不能归因于单一因素。

## 其他检索到的候选

- [SA-UNetv2](https://github.com/clguo/SA-UNetv2)：ISBI 2026 作者实现，主打轻量与精度；本次未在作业上运行。
- [FRetinalNet](https://github.com/SuperXionghaoren/FRetinalNet)：作者发布频域与空间分支模型及 DRIVE 评测实现；本次未在作业上运行。
- [FSE-Mamba](https://pubmed.ncbi.nlm.nih.gov/40743886/)：2025 年论文，提供作者代码地址；本次未在作业上运行。

本地结果以 `outputs/fr_unet_chase_v1/selection.json` 为准。原先 4 张开发验证图已重复用于选型，因此其分数不能当作独立测试成绩。线上效果只以实际 Kaggle 提交返回为准。

复现命令（从上一级目录运行）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u task4_vessels\fr_unet_transfer.py
.\.venv\Scripts\python.exe -X utf8 task4_vessels\verify.py --run fr_unet_chase_v1
# 从最终权重重建预测，不重训
.\.venv\Scripts\python.exe -X utf8 task4_vessels\fr_unet_transfer.py --predict-only
```

重新训练需传入新的 `--run`，避免覆盖已存证实验。
