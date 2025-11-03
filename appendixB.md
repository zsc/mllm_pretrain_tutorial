# 附录 B：工具链与脚手架（配置与模板）

### B.1 开篇与学习目标

本附录旨在将前述章节的理论和策略转化为可执行的蓝图。对于AI科学家和基础设施工程师而言，成功的预训练项目不仅依赖于深刻的理论理解，更依赖于一套健壮、可复现、标准化的工具链和工作流。本章将提供一系列即用型或稍加修改即可部署的配置模板、脚本片段和清单，覆盖从数据抓取到模型发布的整个生命周期。

**学习目标:**

*   掌握各阶段核心工具的具体配置方法。
*   理解如何通过标准化的 Schema 和模板来保证大规模协作的一致性。
*   获得一套用于数据处理、训练、监控、评测和发布的“脚手架”，加速项目启动。
*   预见并规避工具链和工作流中的常见工程问题。

### B.2 数据侧：抓取、ETL 与 Schema 约定

数据是模型的基石，而标准化的数据管理是大规模数据处理的生命线。

#### B.2.1 抓取工具配置（以 `yt-dlp` 为例）

`yt-dlp` 是从 YouTube 等视频平台抓取音视频和字幕的强大工具。以下是一个生产级的推荐配置，通过命令行参数实现：

```bash
# 抓取视频、音频和字幕的推荐脚本
VIDEO_ID="dQw4w9WgXcQ" # 示例视频 ID

yt-dlp \
  --ignore-errors \
  --download-archive ./downloaded_archive.txt \
  --write-sub --write-auto-sub --sub-lang "en,zh-Hans,zh-Hant" \
  --extract-audio --audio-format wav --audio-quality 0 \
  -f 'bestvideo[height<=720]+bestaudio/best[height<=720]' \
  -o "/path/to/raw_data/youtube/%(id)s/%(id)s.%(ext)s" \
  --write-info-json \
  "https://www.youtube.com/watch?v=${VIDEO_ID}"
```

**关键参数解析:**

*   `--ignore-errors`: 批量抓取时跳过不可用视频，保证任务连续性。
*   `--download-archive FILE`: 记录已下载视频ID，避免重复抓取。
*   `--write-sub`, `--write-auto-sub`: 同时抓取官方字幕和自动生成字幕，后者可作为对齐的补充参考。
*   `--sub-lang`: 指定需要的字幕语言。
*   `--extract-audio`: 将音轨分离为单独文件。
*   `--audio-format wav`: 推荐使用无损格式 `wav` 以便后续处理，如果存储压力大，可选用 `flac` 或高比特率 `opus`。
*   `-f '...'`: 格式选择。限制分辨率（如 `720p`）可大幅节约存储和处理成本。
*   `-o "..."`: 标准化的输出路径模板，便于后续 ETL 查找。
*   `--write-info-json`: 导出包含所有元数据（标题、描述、上传者、许可证等）的 `.json` 文件，至关重要。

#### B.2.2 ETL 与数据 Schema 约定

所有原始数据都应经过 ETL (Extract, Transform, Load) 流程，转换为统一格式（如 Parquet），并遵循严格的 Schema。这能极大简化后续处理。

**统一元数据 Schema 模板 (`metadata.json` or Parquet schema):**

```json
{
  "unique_id": "string", // 全局唯一标识符, e.g., "youtube_dQw4w9WgXcQ"
  "source": "string", // 数据来源, e.g., "youtube", "common_crawl", "librivox"
  "source_uri": "string", // 原始链接
  "modalities": ["text", "audio", "video"], // 本条目包含的模态
  "data_paths": {
    "text": "s3://bucket/processed/text/id.txt",
    "audio": "s3://bucket/processed/audio/id.wav",
    "video": "s3://bucket/processed/video/id.mp4",
    "image": "s3://bucket/processed/image/id.jpg"
  },
  "metadata": {
    "title": "string",
    "description": "string",
    "license": "string", // e.g., "CC-BY-3.0", "YouTube Standard License"
    "upload_date": "datetime",
    "duration_seconds": "float",
    "resolution": {"width": "int", "height": "int"},
    "sample_rate_hz": "int",
    "language": "string" // 主要语言 (ISO 639-1 code)
  },
  "quality_scores": {
    "text_perplexity": "float",
    "audio_snr_db": "float",
    "image_aesthetic_score": "float"
  },
  "processing_log": [
    {"stage": "crawl", "timestamp": "datetime", "status": "success"},
    {"stage": "dedup", "timestamp": "datetime", "hash": "string"}
  ],
  "data_card_info": {
      "collection_process": "Automated crawl via yt-dlp v2023.10.13",
      "pii_detection_method": "regex_v1.2"
  }
}
```

### B.3 离散化流水线：音频/视频/图像编码器与 RVQ 配置

离散化是将连续信号转换为 token 序列的关键步骤。其配置直接影响 token 质量和码率。

**音频离散化配置模板 (基于 Encodec):**

```yaml
# audio_tokenizer_config.yaml
model_name: "facebook/encodec_24khz"
target_bandwidth: 6.0 # 码率 (kbps)，6.0 kbps 约等于 75 tokens/秒
sample_rate: 24000 # 必须与模型匹配
num_quantizers: 8 # RVQ 的层数
frame_rate: 75 # 模型输出 token 的速率 (Hz)
audio_chunk_duration_sec: 30.0 # 处理音频的块长度
```

**视频离散化配置模板 (基于 VQ-VAE/MAGVIT):**

```yaml
# video_tokenizer_config.yaml
model_checkpoint: "/path/to/video_vqvae.pt"
input_resolution: [256, 256]
sampling_rate: 4 # 每 4 帧采样 1 帧
tubelet_size: [2, 16, 16] # 时空块 (T, H, W)
codebook_size: 4096 # 码本大小
num_quantizers: 4 # RVQ 层数, 可选
output_tokens_per_second: ~18.75 # (75Hz_audio / 4_frame_sampling)
```

**图像离散化配置模板 (基于 VQGAN):**

```yaml
# image_tokenizer_config.yaml
model_checkpoint: "/path/to/vqgan_f16_16384.ckpt"
resolution: 256
patch_size: 16
latent_shape: [16, 16] # 256 / 16 = 16
codebook_size: 16384
output_tokens_per_image: 256 # 16 * 16
```

### B.4 训练框架：Megatron-LM 配置模板与日志结构

以下是运行一个 10B 级别模型在 256xH100 集群上的 Megatron-LM 启动脚本模板。

**`run_10b_pretrain.sh` 脚本模板:**

```bash
#!/bin/bash

# -- 并行配置 --
TP=4                # 张量并行
PP=8                # 流水线并行
DP=8                # 数据并行 (256 / 4 / 8 = 8)
SP=true             # 开启序列并行以支持长序列

# -- 模型配置 --
NLAYERS=40
NHIDDEN=4096
NHEADS=32
FFN_HIDDEN_SIZE=11008 # SwiGLU 常用 2/3 * 4 * H
SEQ_LEN=8192

# -- 优化器配置 --
LR=1.0e-4
MIN_LR=1.0e-5
LR_DECAY_ITERS=3000000000 # 约 10T tokens
WARMUP_ITERS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0

# -- 训练配置 --
GLOBAL_BATCH_SIZE=1024
MICRO_BATCH_SIZE=1
GAS=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP))) # 梯度累积步数

# -- 混合精度与优化 --
FP8_HYBRID=true       # 启用 FP8
USE_FLASH_ATTN=true
RECOMPUTE_GRANULARITY='full'
RECOMPUTE_METHOD='block'

# -- 数据路径 --
DATA_PATH="/path/to/blended/megatron_data"

# -- 执行命令 --
torchrun --nproc_per_node 8 pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-tokens 10000000000000 \ # 10T tokens
    --lr $LR \
    --min-lr $MIN_LR \
    --lr-decay-style cosine \
    --lr-warmup-iters $WARMUP_ITERS \
    --lr-decay-tokens $LR_DECAY_ITERS \
    --weight-decay $WEIGHT_DECAY \
    --grad-clip-norm $GRAD_CLIP \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --use-flash-attn \
    --sequence-parallel \
    --recompute-granularity $RECOMPUTE_GRANULARITY \
    --recompute-method $RECOMPUTE_METHOD \
    --fp8-hybrid \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --data-path $DATA_PATH \
    --vocab-file /path/to/tokenizer/vocab.json \
    --merge-file /path/to/tokenizer/merges.txt \
    --data-impl mmap \
    --split 990,9,1 \
    --exit-on-missing-checkpoint
```

**日志结构 (JSON Lines):**

为了便于解析和监控，训练日志应输出为 JSON Lines 格式。

```json
{"timestamp": "2024-05-21T10:00:00Z", "iteration": 100, "loss": 2.854, "grad_norm": 0.98, "lr": 9.5e-5, "consumed_tokens": 838860800, "throughput_tps": 3.5e6, "gpu_mem_used_gb": 72.5, "loss_scale": 65536.0}
{"timestamp": "2024-05-21T10:00:10Z", "iteration": 110, "loss": 2.849, "grad_norm": 0.95, "lr": 9.5e-5, "consumed_tokens": 872415232, "throughput_tps": 3.6e6, "gpu_mem_used_gb": 72.6, "loss_scale": 65536.0}
```

### B.5 监控/看板：指标命名与报警门限模板

使用 Prometheus + Grafana 进行监控。一致的指标命名至关重要。

**指标命名约定:**

`project.component.metric_name{label1="value1", ...}`

*   **训练:** `llm_pretrain.training.loss{model="10b", rank="global"}`
*   **吞吐:** `llm_pretrain.training.throughput_tokens_per_second{model="10b"}`
*   **硬件:** `llm_pretrain.hardware.gpu_utilization{node_id="dgx-01", gpu_id="0"}`
*   **数据:** `llm_pretrain.data_pipeline.processed_bytes_total{modality="video"}`

**报警门限模板 (Prometheus `alert.rules.yml`):**

```yaml
groups:
- name: LLMPretrainingAlerts
  rules:
  - alert: TrainingLossStagnation
    expr: abs(avg_over_time(llm_pretrain_training_loss{model="10b"}[1h]) - avg_over_time(llm_pretrain_training_loss{model="10b"}[1h] offset 1h)) < 0.001
    for: 2h
    labels:
      severity: warning
    annotations:
      summary: "训练损失在过去2小时内停滞不前"
      description: "模型 {{ $labels.model }} 的损失值变化小于 0.001，可能已不收敛。"

  - alert: GpuUtilizationTooLow
    expr: avg by (node_id) (llm_pretrain_hardware_gpu_utilization) < 50
    for: 15m
    labels:
      severity: critical
    annotations:
      summary: "节点 {{ $labels.node_id }} GPU 利用率过低"
      description: "GPU 利用率持续低于 50%，可能存在 IO 瓶颈或程序挂起。"

  - alert: GradNormExplosion
    expr: llm_pretrain_training_grad_norm{model="10b"} > 10.0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "梯度范数爆炸"
      description: "模型 {{ $labels.model }} 的梯度范数超过 10.0，训练可能即将发散。"
```

### B.6 评测脚本与用例设计（数据泄漏卫士）

结构化的评测框架能保证结果的可复现性。

**评测任务配置文件 (`eval_task.yaml`):**

```yaml
task_name: "Video_MSRVTT_QA"
eval_class: "VideoQuestionAnswering"
dataset_config:
  path: "s3://eval-datasets/msrvtt/qa_test.json"
  video_root: "s3://eval-datasets/msrvtt/videos/"
metric_config:
  - name: "ExactMatch"
  - name: "BLEU"
prompt_template: |
  <|im_start|>user
  The following are frames from a video. Answer the question based on the video content.
  <vid>{video_placeholder}</vid>
  Question: {question}
  <|im_end|>
  <|im_start|>assistant
  Answer:
```

**数据泄漏卫士 (Canary) 实现思路:**

1.  **创建金丝雀样本**: 从每个评测集中随机抽取一小部分样本（如 20 个），构成一个"金丝雀"集合。
2.  **注入训练集**: 将这些金丝雀样本**原封不动地**混入训练数据中。
3.  **专用评测**: 在每次例行评测时，除了在标准评测集上运行，还要专门在"金丝雀"集合上进行评测。
4.  **告警逻辑**:
    ```python
    def check_leakage(canary_results, standard_results):
        canary_accuracy = canary_results["accuracy"]
        standard_accuracy = standard_results["accuracy"]
        
        # 如果模型在金丝雀样本上表现完美或近乎完美，
        # 而在标准集上表现正常，则极有可能是记忆/泄漏。
        if canary_accuracy > 0.98 and canary_accuracy > standard_accuracy + 0.3:
            return True # Leakage detected!
        return False
    ```

### B.7 发布清单与复现实验记录

#### 发布清单模板 (Markdown)

```markdown
# Model Release Checklist: [Model Name] - v1.0

## Pre-Release
- [ ] 1. **Checkpoint Finalized**: Checkpoint hash verified and stored in a versioned location.
- [ ] 2. **Tokenizer & Encoders**: All tokenizer and discrete encoder versions are documented and packaged.
  - [ ] Text Tokenizer: `qwen-tokenizer @ commit_hash`
  - [ ] Audio Encoder: `encodec_24khz @ config_version_1.1`
  - [ ] ...
- [ ] 3. **Evaluation Complete**: Final evaluation results on all key benchmarks are recorded.
  - [ ] Text PPL: `...`
  - [ ] MMLU: `...`
  - [ ] VideoQA: `...`
- [ ] 4. **Data Leakage Check**: Canary test passed.
- [ ] 5. **Model Card**: `model_card.md` written and reviewed.
- [ ] 6. **Data Card**: `data_card.md` written and reviewed.
- [ ] 7. **License**: `LICENSE` file included and correct.
- [ ] 8. **Inference Code**: Sample inference script is working and tested.

## Release
- [ ] 1. Package all artifacts (checkpoint, configs, cards, code) into a single bundle.
- [ ] 2. Upload to model hub / object storage.
- [ ] 3. Announce release with links and documentation.
```

#### 实验记录模板 (CSV or Markdown Table)

| Experiment ID | Date       | Git Commit | Config Path               | Key Changes                                   | Result Summary                                         | Notes                                           |
|---------------|------------|------------|---------------------------|-----------------------------------------------|--------------------------------------------------------|-------------------------------------------------|
| `exp-10b-lr-1`| 2024-05-20 | `a1b2c3d`  | `configs/10b_base.sh`     | `LR=1e-4`, `Warmup=2000`                        | Loss curve stable, PPL@10k steps = 3.2                 | Baseline run.                                   |
| `exp-10b-lr-2`| 2024-05-22 | `e4f5g6h`  | `configs/10b_lr_halved.sh`| `LR=5e-5`                                     | Slower convergence initially, but loss looks smoother. | Test if lower LR improves stability.            |
| `exp-10b-sp`  | 2024-05-25 | `i7j8k9l`  | `configs/10b_seq_par.sh`  | `SP=true`, `SeqLen=16384`                     | Throughput dropped by 15%, OOM on microbatch > 1.  | Exploring longer context. Needs more tuning.    |

### B.8 本章小结

本附录提供了一系列贯穿多模态大模型预训练全流程的实用工具链模板和脚手架。从 `yt-dlp` 的抓取配置，到标准化的数据 Schema；从离散化流水线的参数设置，到 Megatron-LM 的训练脚本；再到监控报警、评测框架和发布清单。这些模板旨在成为个坚实的起点，帮助团队快速建立起一套规范、高效、可复现的工程实践，从而将主要精力聚焦于模型和数据的核心创新上。

### B.9 常见陷阱与错误 (Gotchas)

1.  **配置地狱 (Config Hell)**: 项目中存在大量配置文件（数据、模型、训练、评测）。
    *   **解决方案**: 将所有配置文件纳入 Git 版本控制。使用单一可信来源（如一个主 YAML 文件）生成不同阶段所需的具体配置，避免手动同步。
2.  **无声的数据错误 (Silent Data Bugs)**: ETL 脚本中的一个微小 bug 可能污染 TB 级的训练数据，而训练过程可能不会立即报错。
    *   **解决方案**: 严格执行 Schema 校验。在 ETL 的每一步都进行数据抽样和断言（assert），例如检查 token 序列长度、特殊符号比例等是否在预期范围内。
3.  **环境不一致导致复现困难**: 离散化环境中的 `torch` 版本与训练环境不一致，或 tokenizer 版本不同，导致模型输入不匹配。
    *   **解决方案**: 使用容器化技术（如 Docker, Singularity/Apptainer）封装每个阶段的环境，确保从数据处理到训练推理的完全一致性。
4.  **监控盲点**: 只监控了损失函数，却忽略了 GPU 利用率、网络 IO 或数据加载器吞吐。
    *   **解决方案**: 在项目第一天就建立全面的监控看板。一个无法观测的指标等于不存在。训练早期的性能瓶颈往往不在计算，而在数据或通信。
5.  **“本地能跑”陷阱**: 在单机上测试通过的脚本，在 256 卡的集群上可能因为竞态条件、文件系统压力或网络延迟而频繁失败。
    *   **解决方案**: 尽早（即使是用小模型）在目标规模的集群上进行端到端的“冒烟测试”，验证整个工作流的健壮性。
