# 作业二：人脸表情分类

本项目使用 PyTorch 在本机 NVIDIA GPU 上训练六分类 ResNet18/ResNet34，并生成 Kaggle 要求的 `ID,Emotion` CSV。实验报告暂不包含在本项目交付范围内。

## 2026-10-18：更换主干网络实验

本轮新增 **EfficientNet-B0**，使用 [torchvision 官方 ImageNet 权重](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)，将分类头替换为六分类。它与已有 ResNet 的特征提取结构不同。核心阅读顺序仍为 `model.py`（网络与图像预处理）、`train.py`（训练、验证、EMA和选取权重）、`predict.py`（输出类别概率与提交文件）。`efficientnet_study.py` 负责将这些步骤按固定计划串起来。

训练使用160像素输入、batch128、12轮、AdamW、初始主干学习率0.0003、分类头学习率为主干的3倍、Mixup和EMA。验证划分仍为24,034张训练图、4,241张验证图。新模型架构、预训练权重及输入大小与原方案不同，因此这是**新训练方案的比较，不是只改网络结构的单因素实验**。原始图片仍为48×48灰度图，放大输入不会创造额外图像信息。

`outputs/late_20261018/efficientnet/plan.json` 在训练前固定了5个候选：原四模型、新模型单独、原四模型加新模型、原三模型加新模型、种子44模型加新模型。均使用原图/水平翻转的概率平均，融合时等权平均；至少比原四模型多答对43张验证图，才生成新的候选提交。这个门槛约为1.01个百分点，是实际筛选标准，不代表统计显著性或线上提升保证。验证集已用于多次方案选择，不能当作未参与选型的独立测试集。

本轮的预备尝试有明确记录：中心44×44裁剪及翻转使验证正确数从2,973增至2,983（70.10%→70.34%），但用户反馈对应 `submission_center_tta.csv` 的Kaggle成绩为 **0.719**，低于原来的 **0.722**，因此保留旧方案作为当前线上基准。112/160短程微调对照在112组完成2轮后，因用户希望优先进行更大幅度的模型改进而中止；160组没有开始，不能据此得出分辨率对比结论。详见 `outputs/late_20261018/online_feedback.json` 和 `resolution/status.json`。

EfficientNet实验当前状态：**因剩余时间预算在完成第8/12轮后停止**，每个已完成轮次都使用完整的24,034张训练图。当前最佳为第7轮RAW权重，不含TTA的验证准确率为**65.9986%（2,799/4,241）**。12轮原计划没有完成，5个候选的翻转TTA/融合比较尚未计算，也未生成EfficientNet的新提交；不能把已计划的比较写成实验结果。停止记录见 `outputs/late_20261018/efficientnet/status.json`。

**ResNet50已启动16轮训练**，正式配置为112像素输入、batch128、保留标准初始最大池化（`no_maxpool=false`），使用 [torchvision 官方 `ResNet50_Weights.IMAGENET1K_V2` 权重](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)，分类头改为六分类。其余采用上述AdamW、Mixup、EMA及相同验证划分。`outputs/late_20261018/resnet50/plan.json` 在训练前固定了对应的5个候选和增加43张正确预测的门槛。此次网络、预训练与输入配置合在一起构成新的实用方案，不声称是单因素消融。

考虑截止时间，ResNet50默认路径在达到门槛后直接使用选中的验证训练权重预测测试集，**新模型只使用24,034张训练图，不执行28,275张图的全量重训**；参与融合的旧模型仍使用已有全量权重。每个模型实际训练范围、最佳轮次和RAW/EMA来源会写入独立审计。尚未完成训练及选型，不能据此承诺新的线上分数。实际启动命令为：

```powershell
.\.venv\Scripts\python.exe -u resnet50_study.py --stage all --epochs 16 --image-size 112
# 完成训练和候选比较后生成真实图表：
# .\.venv\Scripts\python.exe plot_resnet50_results.py
```

以下命令可继续未完成的EfficientNet原计划，或只绘制已完成8轮的曲线；完整方案审计应在实际完成比较和预测之后运行：

```powershell
.\.venv\Scripts\python.exe -u efficientnet_study.py --stage all --epochs 12 --image-size 160
.\.venv\Scripts\python.exe plot_efficientnet_results.py
# 完成方案比较和提交生成之后再执行：
# .\.venv\Scripts\python.exe audit_efficientnet.py
```

`verify_efficientnet.py` 检查旧ResNet构建及权重加载行为不变、新网络六分类输出和优化器参数分组；`audit_efficientnet.py` 检查方案选择、模型来源及提交文件。速度测试脚本仅用于估算耗时，其随机初始化权重从未用于正式训练或提交。`efficientnet_deadline_submit.py` 是截止时间紧张时的备用路径：它使用经过验证的EfficientNet权重直接预测测试集，并明确标记新模型只使用24,034张训练图，不会伪称已经完成28,275张图的全量重训。

## 2026-10-16 随机种子与随机擦除对照

**本轮已于2026-10-17完成全量训练和最终审计。提交文件：`outputs/seed_erasing_20261016/submission_improved.csv`，共7,065条预测。** 新种子模型从第9轮断点续训至第13轮，最终与原有三个全量模型等权融合。用户随后反馈本轮Kaggle成绩仍为 **0.722**；该成绩来自用户反馈，未由程序独立查询。上一轮线上0.7220的第二版文件也保持不变。

`audit.json` 确认提交ID与模板逐行一致、标签合法、CSV与概率最大值一致，四个最终模型均完成全量训练且权重来源与选型一致。新提交SHA256：`b889290cc001ca7e5c3d8c375693b85e183656f3f81c63ac3cc843a1788ce80a`。仅重新生成本轮预测可运行 `.\.venv\Scripts\python.exe -u seed_erasing_study.py --stage final`，已完成的训练会复用。

基准为上一轮去掉初始最大池化的 ResNet18，固定同一份分组划分（训练24,034张、验证4,241张），保持网络、输入112、batch128、学习率、18轮计划、Mixup、EMA等设置一致：

- **只换随机种子**：43 → 44，随机擦除仍关闭。
- **只加随机擦除**：仍用种子43，每张训练图以25%概率擦除2%～8%面积的矩形；长宽比在0.5～2.0间按对数均匀采样，填充值为该图增强后的灰度均值。擦除发生在归一化及Mixup之前；验证和测试不擦除。像素取整会使实际面积有小幅偏差。

两组分别训练18轮，按不含TTA的验证准确率选择最佳raw/EMA权重，再统一评估水平翻转TTA。原模型最佳为第11轮EMA，新种子模型为第13轮EMA，擦除模型为第12轮EMA。`plan.json` 在新模型评估前固定了9个候选，均使用等权平均，没有搜索连续融合权重。

| 方案（全部含水平翻转TTA） | 验证Accuracy | 正确数 / 4,241 |
|---|---:|---:|
| 单模型基准：种子43、无擦除 | 68.76% | 2,916 |
| 只换种子44 | 69.51% | 2,948 |
| 只加轻量擦除，种子43 | 69.32% | 2,940 |
| 种子43与44两个模型平均 | 69.98% | 2,968 |
| 基准模型与擦除模型平均 | 69.37% | 2,942 |
| 上轮三模型融合（用户反馈线上0.7220） | 69.63% | 2,953 |
| 上轮融合 + 种子44（选中） | **70.10%** | **2,973** |
| 上轮融合 + 擦除模型 | 70.05% | 2,971 |
| 上轮融合 + 两个新模型 | 70.01% | 2,969 |

本次种子变化和擦除分别使单模型多答对32张、24张；这说明它们在本次划分中有效，不能推断某个种子普遍更好。最终选中四模型融合，比上轮多答对20张，即**0.47个百分点**；逐样本比较为原错新对66张、原对新错46张。四模型只比加入擦除的另一组四模型多对2张，差异很小；同时加入两个新模型没有继续改善，所以最终不使用擦除权重。

最终方案复用原有三个全量模型，仅将种子44模型在全部28,275张有标签图上从ImageNet初始化重训13轮，保持18轮余弦学习率计划。每个模型先平均原图与水平翻转图的概率，再对四个模型等权平均。**70.10%是重复用于方案选择的本地验证成绩，不是全量权重的独立验证成绩，也不是新的Kaggle成绩。**

实际对比数据、图片及审计分别见 `selection.json`、`comparison.csv`、`comparison.png`、`validation_audit.json`。各组保存完整 `config.json`、`history.csv`、最佳权重及训练曲线。`erasing_checks.json` 记录擦除实现检查：关闭擦除时与旧源码的输出和随机数状态逐位一致；验证不受影响；遮挡独立且符合面积和填充设定。`source_snapshot/` 保存源码及SHA256清单。

```powershell
# 使用新目录复现本轮两项对照、固定候选比较及有提升时的全量重训。
.\.venv\Scripts\python.exe seed_erasing_study.py --stage all --output-dir outputs/seed_erasing_reproduction
.\.venv\Scripts\python.exe audit_seed_erasing.py --output-dir outputs/seed_erasing_reproduction
.\.venv\Scripts\python.exe plot_seed_erasing.py --output-dir outputs/seed_erasing_reproduction
```

这些命令需要现有 `data/cache.npz` 及上一轮模型/预测；不会重新训练被复用的三个旧模型。训练时未开启严格确定性CUDA算法，相同配置重新运行可能有细微数值差异。验证集划分始终固定，未使用测试标签或按测试重复图替换预测。

## 2026-10-16 改进实验

**第二版已完成全量训练和独立审计。上传文件：`outputs/improvement_20261016/submission_improved.csv`，共7,065条预测。** 审计确认ID顺序、标签、概率、实际权重来源及全量训练记录一致，原提交文件保持不变。

用户反馈第一版 Kaggle 成绩为 **0.70**。新增两组结构对照：

- **保留表情细节**：ResNet18 去掉最前面的最大池化层；112×112 输入下，最后一组卷积的特征图从 4×4 增为 7×7。卷积权重仍从官方 ImageNet 预训练模型初始化。
- **增加网络深度**：使用标准 ResNet34，其余训练参数与旧 Mixup 方案一致。结构与预训练权重见 [torchvision 官方说明](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html)。

两组实验继续使用原来的分组验证集、seed=43、输入112、batch128、18轮学习率计划、Mixup、Dropout及权重衰减设置。每组根据不含 TTA 的验证准确率选取最佳权重，再统一比较水平翻转 TTA 的结果及简单等权概率平均。这个验证集用于方案选择，不能当作从未参与选型的最终测试集。

实际验证结果如下，均为同一组 4,241 张图片：

| 方案（均含水平翻转 TTA） | 验证 Accuracy |
|---|---:|
| 第一版提交方案：基础模型与旧 Mixup 模型平均 | 68.69% |
| 旧 Mixup ResNet18 | 68.59% |
| 保留细节的 ResNet18 | 68.76% |
| 标准 ResNet34 | 67.86% |
| 旧 Mixup ResNet18 + 保留细节的 ResNet18 | 69.56% |
| 上述两个 ResNet18 + ResNet34，三模型等权平均（选中） | **69.63%** |

选中方案答对 **2,953/4,241** 张，原方案答对 2,913 张，增加 40 张，即 **0.94 个百分点**。三模型比最好的两模型仅多答对 3 张（0.07 个百分点），这个小差异不应夸大。此次数据支持保留更多空间细节和模型互补，不能据此得出“网络越深，单模型越好”的结论。

逐样本比较：原来答错、现在答对159张；原来答对、现在答错119张。提升主要出现在 Angry、Fear、Neutral，Sad 召回率略降，Surprise 不变。方法对比与各类召回率图见 `outputs/improvement_20261016/comparison.png`，具体值保存在选型及审计 JSON 中。

最终配置使用旧 Mixup ResNet18（全量训练12轮）、保留细节的 ResNet18（11轮）与 ResNet34（9轮）。三者都保留18轮学习率计划，先平均每个模型对原图与水平翻转图的概率，再等权平均三个模型的概率，取最大概率对应的类别。仍是常规残差网络、数据增强和概率平均，不涉及伪标签或复杂自定义损失函数。具体选型见 `outputs/improvement_20261016/selection.json`，单变量配置差异见 `controlled_changes.json`。

**69.63% 是本地选型验证成绩。用户随后反馈这版 Kaggle 成绩为 0.7220；该线上结果来自用户反馈，未由程序独立查询。不能用验证提升直接推算后续线上分数。**

改进实验输出位于 `outputs/improvement_20261016/`，原有提交 `outputs/submission.csv` 保留。完整流程由 `improve.py` 执行：

```powershell
# 在新目录重新训练、比较，再按选中的轮次全量重训并生成 CSV。
.\.venv\Scripts\python.exe improve.py --stage all --output-dir outputs/improvement_reproduction
```

`--stage validation` 只运行对照与选型，`--stage final` 根据已有的 `selection.json` 全量重训并输出 `submission_improved.csv`。脚本会核对复用目录的数据、训练参数和 checkpoint；遇到未完成的训练，请先用 `train.py --resume` 搭配原参数恢复该训练，再继续流程。

## 比赛约定

| 标签 | 类别 | 中文 | 训练图片数 |
|---|---|---|---:|
| 0 | Angry | 愤怒 | 3963 |
| 1 | Fear | 恐惧 | 4097 |
| 2 | Happy | 快乐 | 7192 |
| 3 | Sad | 悲伤 | 4862 |
| 4 | Surprise | 惊讶 | 3202 |
| 5 | Neutral | 中性 | 4959 |

评分指标为 Accuracy。依据用户提供的规则截图，必须使用深度学习，个人完成，队伍名称为个人学号，每天最多提交 5 次。截图注明截止时间为 2026 年 10 月 18 日 23:59。

**不要用文件夹的字母顺序生成提交标签**：Neutral、Sad、Surprise 的字母排序与竞赛编号不同。

## 已核验的数据

- 来源：用户下载的 `D:\Downloads\2026task2.zip` 和 `D:\Downloads\submission.csv`。
- 28,275 张有标签训练图片，7,065 张无标签测试图片，全部 48×48 单通道灰度 JPEG。
- 全部图片解码成功，压缩包 CRC 检查通过。
- 数据准备直接读取 ZIP 并缓存为 NumPy 数组，无需解压几万张小图。
- 固定 seed=42，按解码像素 SHA256 分组后近似分层划分：24,034 张训练，4,241 张验证。
- 重复像素组（包括标签冲突的组）不会分散到训练集和验证集。训练/验证完全重复图片交集为 0。
- 训练集有 1,000 个重复像素组，其中 39 组标签冲突；保留原始标签。训练/测试也存在像素重合，仅做数据审计；预测流程不使用测试标签，也不按重复匹配替换模型结果。
- 详细信息见 `data/audit.json`；原始 CSV 全为 0，只是格式模板。

## 本机环境

已检测到 NVIDIA GeForce RTX 4060 Laptop GPU（8GB 显存）。项目 Python 环境位于 `.venv`，与基础环境中的 CPU 版 PyTorch 分开。

如需重新配置，在项目目录打开 PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

首次需从 [PyTorch 官方源](https://pytorch.org/get-started/previous-versions/) 下载 CUDA 12.8 版 PyTorch。项目固定 PyTorch 2.11.0、torchvision 0.26.0；GPU 驱动支持时无需另外安装完整 CUDA Toolkit。

## 第一版选型结果

两个方案均使用同一份 4,241 张验证图片，训练时不接触这些图片。每个方案训练 18 轮，再按验证准确率取最佳 checkpoint。结果均来自实际 GPU 运行：

| 方案 | 本地验证 Accuracy |
|---|---:|
| 基础模型，第 7 轮 EMA | 66.89% |
| 基础模型 + 水平翻转 TTA | 67.34% |
| Mixup 方案，第 12 轮 EMA | 67.65% |
| Mixup 方案 + 水平翻转 TTA | 68.59% |
| 两模型概率平均 | 68.22% |
| 两模型概率平均 + 水平翻转 TTA（选定） | **68.69%** |

Mixup 方案同时调整了 Dropout、权重衰减和随机种子，因此上表不是单一变量的消融实验。全部比较见 `outputs/selection.json`。

最终方案使用全部 28,275 张有标签图重新训练基础模型 7 轮、Mixup 模型 12 轮，保持原来的 18 轮余弦学习率计划。测试时分别平均原图和水平翻转图的预测概率，再平均两个模型的概率并取最大类别。

**68.69% 是选型模型的验证分数，不是全量重训权重的独立验证分数，也不是 Kaggle 测试成绩。** 全量重训模型已见过原验证图片，程序禁止用这些权重再报告独立验证结果。

## 复现第一版预测

所有命令均在本项目目录执行。

1. 准备数据（现有缓存可直接复用）：

```powershell
.\.venv\Scripts\python.exe prepare_data.py --zip D:\Downloads\2026task2.zip --submission D:\Downloads\submission.csv --output-dir data
```

2. 从官方 ImageNet 初始化重新执行选定的全量训练和预测：

```powershell
powershell -ExecutionPolicy Bypass -File .\train_final.ps1 -OutputRoot outputs/reproduction -Submission outputs/reproduction_submission.csv
```

首次实际运行的模型位于 `outputs/final_models/`，无需重复训练即可生成提交。上述复现命令使用新目录；程序会拒绝覆盖已有训练目录。CUDA 默认使用固定种子及高吞吐算法，跨硬件或重新运行的末位结果不保证逐位相同；可通过 `train.py --deterministic` 请求较严格的确定性算法。

3. 只用现有最终模型重新生成 CSV 并检查格式：

```powershell
.\.venv\Scripts\python.exe predict.py --data data/cache.npz --checkpoints outputs/final_models/baseline/final.pt outputs/final_models/mixup/final.pt --tta --device cuda --output outputs/submission.csv
.\.venv\Scripts\python.exe verify_submission.py outputs/submission.csv --json outputs/submission.check.json
```

## 重做第一版验证实验

以下为本次选型的训练参数。已有目录不能覆盖，重新实验请更换输出目录：

```powershell
.\.venv\Scripts\python.exe train.py --output-dir outputs/resnet18_main --epochs 18 --batch-size 128 --image-size 112 --lr 0.0003 --seed 42 --device cuda
.\.venv\Scripts\python.exe train.py --output-dir outputs/resnet18_mixup_s43 --epochs 18 --batch-size 128 --image-size 112 --lr 0.0003 --weight-decay 0.001 --label-smoothing 0.05 --dropout 0.3 --seed 43 --mixup-alpha 0.2 --mixup-prob 1 --device cuda
.\.venv\Scripts\python.exe predict.py --checkpoints outputs/resnet18_main/best.pt outputs/resnet18_mixup_s43/best.pt --split val --tta --output outputs/validation_ensemble.csv
```

输入灰度图转成三个相同通道，采用 ImageNet 归一化与官方预训练权重。训练包含适度旋转、位移、缩放、水平翻转和亮度/对比度变化；验证不做随机增强。使用 AdamW、混合精度和 EMA。Mixup 的训练准确率按两个标签的混合权重计算，不能直接与普通训练准确率比较。

恢复中断训练时使用 `--resume outputs/resnet18_main/last.pt`，并保持原训练参数一致。正式训练结果目录不可直接覆盖，新实验应使用新的 `--output-dir`。

`--full-train` 使用所有训练图并生成 `final.pt`；`--stop-after-epochs K` 在第 K 轮保存后退出，但不会改变 `--epochs` 指定的余弦学习率计划。恢复到同一目录时允许提高停止轮数，其他训练参数必须匹配。

## 文件与结果

- `prepare_data.py`：检查数据、固定标签、重复图片分组划分、生成缓存。
- `model.py`：模型、GPU 批量预处理、分类指标。
- `train.py`：训练、验证、最佳权重与断点保存。
- `predict.py`：验证/测试推理、水平翻转 TTA、可选多模型概率平均。
- `verify_submission.py`：提交文件的列、行、ID 顺序及标签范围检查。
- `train_final.ps1`：按验证选定配置全量重训、生成并检查提交。
- `improve.py`：运行两组结构对照、比较固定候选组合、全量重训选中的方案。
- `audit_improvement.py`：独立核验改进版的提交、概率、模型来源和验证提升。
- `plot_improvements.py`：从真实选型结果绘制方法比较及各类召回率图。
- `plot_results.py`：从真实训练日志绘制曲线及验证混淆矩阵。
- `outputs/*/history.csv`：每轮训练/验证曲线数据。
- `outputs/*/best_metrics.json`：最佳验证准确率、混淆矩阵、各类召回率。
- `outputs/*/config.json`：训练配置及环境记录。
- `outputs/final_models/*/final.pt`：全量重训的最终推理权重。
- `outputs/environment.json`：实际运行环境版本。
- `outputs/submission.csv`：第一版预测文件，用户反馈线上成绩0.70，原文件保留。
- `outputs/submission.json`：第一版所用模型、TTA 设置和各类预测数量。
- `outputs/submission.check.json`：第一版提交格式检查结果。
- `outputs/improvement_20261016/submission_improved.csv`：本次改进版预测文件。
- `outputs/improvement_20261016/audit.json`：改进版预测、权重来源、全量训练及验证对比的独立审计。
- `outputs/improvement_20261016/source_snapshot/`：本次训练与推理代码快照及 SHA256 清单。

本项目不会自动上传 Kaggle。线上测试集准确率需要上传后由 Kaggle 返回，不能用本地验证准确率代替。
