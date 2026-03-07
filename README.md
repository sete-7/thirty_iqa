# 多源图像评价数据集：自动化构建与校准流水线

一套端到端的 Python 流水线，自动化构建高质量图像评价训练数据集。整合了 **机器专家打分**、**LLM 推理分析** 和 **人类偏好标注**，最终输出统一的可训练数据格式。

## 项目结构

```
thirty_iqa/
├── data_loader.py            # 从本地文件夹或 HuggingFace JSONL 加载图片
├── feature_extractor.py      # DINOv2 图像特征 + CLIP 文本特征提取（含断点续传）
├── expert_scorers.py         # 串行专家打分：UniPercept / Grounding-IQA / HPSv3 / SpatialScore
├── cot_generator.py          # 方差争议检测 → LLM 生成 <think> + <bbox> 结构化 CoT
├── data_filter.py            # 去除首尾各 20%，保留中间 60% 高质量难例
├── app_gradio.py             # Gradio 二选一盲测标注界面 → human_votes.csv
├── mos_calibrator.py         # Thurstone 模型 MOS 校准（choix 库）
├── dataset_packager.py       # 按 Image ID 对齐 → 分离输出 Basic / Reasoning 数据集
├── visualization.py          # 分数分布直方图 + 箱线图
├── requirements.txt          # Python 依赖
└── README.md                 # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意：** `hpsv3` 可能需要从源码安装。美学/空间专家为占位符，需替换为实际工具。

### 2. 准备数据

将测试图片放入文件夹，或准备包含 `image_path` 和 `prompt` 字段的 JSONL 文件：

```jsonl
{"image_path": "images/001.jpg", "prompt": "一只猫坐在沙发上"}
{"image_path": "images/002.jpg", "prompt": "群山上的日落"}
```

### 3. 逐步运行流水线

#### 步骤 1 — 提取 DINOv2 图像特征 + CLIP 文本特征

```python
from data_loader import get_dataloader
from feature_extractor import process_features_with_checkpointing

data = get_dataloader('local', './test_images')   # 或 ('jsonl', 'metadata.jsonl')
process_features_with_checkpointing(data, output_jsonl="features.jsonl")
```

#### 步骤 2 — 专家模型打分（串行执行，自动清显存）

```python
from expert_scorers import process_scoring_with_checkpointing

process_scoring_with_checkpointing(data, output_jsonl="expert_scores.jsonl")
```

**当前专家模型：**
| 模型 | 评估维度 |
|---|---|
| UniPercept | 感知质量 |
| Grounding-IQA | 区域质量 + 缺陷定位 |
| HPSv3 | 语义 / 图文匹配 |
| SpatialScore | 空间布局理解 |

#### 步骤 3 — 过滤中间 60% 难例

```python
from data_filter import filter_data_middle_60

filter_data_middle_60("expert_scores.jsonl", "filtered_scores.jsonl")
```

#### 步骤 4 — 为争议图片生成结构化 CoT

先设置 API 密钥：

```bash
set OPENAI_API_KEY=sk-...
set OPENAI_BASE_URL=https://api.openai.com/v1
set LLM_MODEL_NAME=qwen-vl-max
```

```python
from cot_generator import process_cot_generation

process_cot_generation("expert_scores.jsonl", "train_reasoning_data.jsonl", variance_threshold=1.0)
```

LLM 输出格式示例：
```xml
<think>该图片整体构图较好，但右下角模糊 <bbox>[0.65,0.70,0.95,0.98]</bbox>...</think>
<summary>图片质量中等偏上，存在局部模糊和色彩伪影。</summary>
<final_score>62</final_score>
```

#### 步骤 5 — 人类偏好标注（Gradio 界面）

准备 `app_pairs.jsonl` 文件：

```jsonl
{"prompt": "未来城市", "img_left": "img_a.jpg", "img_right": "img_b.jpg"}
```

启动服务：

```bash
python app_gradio.py
```

打开 `http://localhost:7860`，投票结果保存在 `human_votes.csv`。

#### 步骤 6 — MOS 校准

```python
from mos_calibrator import calibrate_mos_from_votes

calibrate_mos_from_votes("human_votes.csv", "mos_calibration_data.json")
```

#### 步骤 7 — 打包最终数据集（自动分离）

```python
from dataset_packager import package_final_dataset

package_final_dataset(
    features_jsonl="features.jsonl",
    scores_jsonl="expert_scores.jsonl",
    cot_jsonl="train_reasoning_data.jsonl",
    mos_json="mos_calibration_data.json",
    output_basic="Dataset_basic.jsonl",
    output_reasoning="Dataset_Reasoning.jsonl",
    variance_threshold=1.0
)
```

#### 步骤 8 — 可视化数据集健康度

```bash
python visualization.py
```

图表保存在 `plots/score_histograms.png` 和 `plots/score_boxplots.png`。

## 断点续传

`feature_extractor.py`、`expert_scorers.py` 和 `cot_generator.py` 均支持 **每 100 张图片自动保存进度**。如果运行中断，重新运行同一命令即可——已处理的图片会自动跳过。

## 输出文件说明

| 文件 | 说明 |
|---|---|
| `features.jsonl` | 每张图的 DINOv2 CLS 特征 + CLIP 文本特征 |
| `expert_scores.jsonl` | 四个专家模型的独立打分 |
| `filtered_scores.jsonl` | 去除极端值后的中间 60% 数据 |
| `train_reasoning_data.jsonl` | 高争议图片的结构化 CoT 分析 |
| `human_votes.csv` | 人类两两偏好投票 |
| `mos_calibration_data.json` | 校准后的绝对 MOS 分数 |
| `Dataset_basic.jsonl` | **低冲突基础训练集** |
| `Dataset_Reasoning.jsonl` | **高冲突推理训练集（含 CoT）** |
| `plots/` | 分数分布直方图 & 箱线图 |
