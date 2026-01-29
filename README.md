
## 🐥 Overview

**CECAR** is a research framework for **Cache & Expert Co-Aware Routing Accelerates On-Device Inference of MoE LLMs** 

Modern on-device MoE inference faces fundamental limitations due to constrained memory capacity and expert loading latency. Through systematic analysis, we identify three key observations:

 1. **Cache hit rates are fundamentally bounded**, even under Belady’s optimal replacement policy.
 2. **These upper bounds are insufficient** to meet the latency requirements of practical on-device inference.
 3. **Cache-aware approximate expert routing** can significantly improve cache locality with **minimal semantic distortion**.

Based on these observations, we propose CECAR (Cache & Expert Co-Aware Routing)—a MoE inference system that jointly optimizes expert routing and cache management.

CECAR leverages an **ML-based cache policy** that predicts the **expected reuse distance of each expert**, and dynamically prioritizes routing and execution toward experts with smaller predicted reuse distances.
This co-aware design enables higher effective cache hit rates and reduced expert loading overhead, making low-latency on-device MoE inference feasible. 

This repository provides a **complete experimental pipeline** including:

- Model-specific build and data preparation
- Training FFN-based eviction models
- Cache-aware inference with routing and bonus strategies
- Large-scale evaluation in simulation mode

CECAR currently supports the **following three MoE LLMs**: 

`Qwen3_30B_A3B`, `DeepSeek_v2_Lite_Chat`, `OLMoE_1B_7B_0125_Instruct`

All scripts (build, train, inference, evaluation) are designed to work consistently across these models.

CECAR supports two learned eviction models as ML-based cache policies:
+ **ML_CECAR — the proposed cache policy** in the CECAR paper. It is designed to work synergistically with the co-aware routing mechanism to optimize on-device MoE inference performance.
+ **ML_FlashMoE — a reference/baseline learned eviction model** used for comparison. It is not part of the core CECAR proposal but serves to contextualize ML-based eviction performance against CECAR’s policy. [📋FlashMoE Paper](https://www.arxiv.org/abs/2601.17063)

---
## ⚡ Quick Start (Using Pretrained Eviction Models)
For fast experimentation, **CECAR provides pretrained FFN-based eviction models**.

You do NOT need to train eviction models yourself unless you want to reproduce or extend the training results.

After running the **build** step, you can directly proceed to:
+ **Inference**
+ **Evaluation**

using the pretrained **ML_CECAR / ML_FlashMoE** eviction models.

---

## 📦 Install CECAR from Source & Data Preparation

### 💻 Environment Setup

We recommend using **conda** to manage the Python environment.

```
conda create --name CECAR python=3.10 -y
conda activate CECAR
```

### 🛠️ Install CECAR from Source

```
git clone https://github.com/malmong/CECAR
cd CECAR

chmod +x build
chmod +x train
chmod +x inference
chmod +x evaluation
```
+ `git clone` downloads the full CECAR source code.
+ The four shell scripts (`build`, `train`, `inference`, `evaluation`) are the **main entry points** of the framework.
+ `chmod +x` enables execution permission for each script.


Build model-specific assets:

```
./build --model_name {model_name}
```

The `build` script prepares **model-specific assets and dependencies**, including:
+ Huggingface login
+ Downloading required model checkpoints and tokenizer files
+ Installing or verifying model-dependent Python requirements
+ Generating intermediate metadata used by training, inference, and evaluation

Supported `model_name` values:
```
Qwen3_30B_A3B
DeepSeek_v2_Lite_Chat
OLMoE_1B_7B_0125_Instruct
```

You must run `build` **at least once** before running `inference` or `evaluation`.

---
## 🧠 Train FFN Eviction Model

```
./train \
  --model_name {model_name} \
  --mode {mode} \
  --train_ffn_model {train_ffn_model} \
  --eval_model {eval_model}
```

The `train` script is used to **train and/or evaluate ML-based cache eviction models**.
+ CECAR provides **pretrained FFN-based eviction models**, so **this step can be skipped** if you only want to run inference or evaluation.
+ Training is required only if you want to:
  + Reproduce training results
  + Train eviction models on new settings
  + Compare different ML-based cache policies

`ML_CECAR` and `ML_FlashMoE` can be trained and evaluated for cache hit-rate prediction performance.

### Options

```
model_name        :  Qwen3_30B_A3B | DeepSeek_v2_Lite_Chat | OLMoE_1B_7B_0125_Instruct 
mode              :  train_eval | train_only | eval_only
train_ffn_model   :  ML_CECAR | ML_FlashMoE
eval_model        :  both | ML_CECAR | ML_FlashMoE
```

---
## 🚀 Inference
```
./inference \
  --model {model_name} \
  --test-task {task} \
  --task-num 0 \
  --cache-policy {cache_policy} \
  --mode {mode} \
  --bonus-strategy {bonus_strategy}
```

The inference script runs **actual MoE inference with cache-aware routing**.

Enables efficient measurement of:
+ Cache hit rate
+ Prefill & Decoding speed
+ Generated Token

CECAR supports the following benchmark tasks: `humaneval`, `mbpp`, `gpqa`, `math500`
+ Each task provides **5 predefined samples**
+ The sample index is selected using `--task-num` (`0 ~ 4`)

### Options

```
model_name       : Qwen3_30B_A3B | DeepSeek_v2_Lite_Chat | OLMoE_1B_7B_0125_Instruct 
task             : humaneval | mbpp | gpqa | math500
task-num         : 0 | 1 | 2 | 3 | 4
cache-policy     : lru | lfu | lifo | ML_CECAR | ML_FlashMoE
mode             : des | odp | none
bonus-strategy   : cecar | mocce | none
```

**Routing / Bonus Interaction Rules**
+ **DES** and **ODP** are routing-only mechanisms so `bonus_strategy` is automatically set to `none` [📋DES & ODP Paper](https://openreview.net/forum?id=hheFYjOsWO)
+ **MOCCE** is a bonus-based cache-aware routing strategy  [📋MOCCE Paper](https://openreview.net/forum?id=ul4W26KEKz)
  + Promotes experts already resident in cache
  + Does not modify eviction decisions

### Custom Prompt Inference

You can bypass benchmark datasets and run inference with **your own prompt**:
```
./inference \
  --model {model_name} \
  --cache-policy {cache_policy} \
  --mode {mode} \
  --bonus-strategy {bonus_strategy} \
  --input {your own prompt}
```

---
## 📊 Evaluation
```
./evaluation \
  --simulation_model {model_name} \
  --tasks {task} \
  --cache_policy {cache_policy} \
  --mode {mode} \
  --bonus_strategy {bonus_strategy} \
  --limit 1 \
  --limit_start 0
```

The `evaluation` script runs **large-scale evaluation in simulation mode**, where:
+ Cache behavior is simulated with **virtual cache**
+ All experts are assumed to reside on device



### Options
```
model_name       : Qwen3_30B_A3B | DeepSeek_v2_Lite_Chat | OLMoE_1B_7B_0125_Instruct 
bonus_strategy   : none | random | constant | lfu | lru | mocce | const | cecar
tasks            : humaneval | mbpp | gpqa | math500
cache-policy     : lru | lfu | lifo | ML_CECAR | ML_FlashMoE
mode             : des | odp | none
bonus-strategy   : cecar | mocce | none
```
#### limit / limit_start

+ `limit_start`: starting index

+ `limit`: number of evaluated samples

If both options are omitted, evaluation is performed on the **entire dataset**.


## Notes

+ All experiments are fully configurable via CLI.
+ Pretrained FFN-based eviction models are provided, so training can be skipped for most use cases.
+ The framework cleanly separates cache policy, routing strategy, and bonus mechanism, enabling controlled ablation studies.
+ For fine-grained control or custom experimentation, users may directly run the underlying Python scripts instead of the provided shell wrappers





