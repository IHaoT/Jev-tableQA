# GraphOTTER 本地复现记录

复现目标：在 macOS (arm64) 上跑通 GraphOTTER 的完整推理链路
（LLM 选初始单元格 → 稠密/稀疏检索 → 图上迭代 Thought/Action/Observation → 最终答案 → EM 评测）。

## 1. 环境

- Python **3.9.6**，独立虚拟环境 `.venv`（`uv venv --python 3.9 .venv`）。
- 依赖锁定在 [requirements-lock.txt](requirements-lock.txt)，版本取自本机已验证可运行的
  conda 环境 `graphotter`：

| 组件 | 版本 |
|---|---|
| torch | 2.2.2 |
| sentence-transformers | 2.2.2 |
| transformers | 4.29.2 |
| faiss-cpu | 1.7.4 |
| numpy | 1.26.4 |
| openai | 2.45.0 |

安装：

```bash
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

## 2. 数据

| 数据集 | 路径 | 规模 | 说明 |
|---|---|---|---|
| HiTab | `dataset/hitab/test_samples.jsonl` + `dataset/hitab/raw/*.json` | 1584 题 / 538 张表 | 原始代码仓库的 `raw/` 只有一个占位文件；raw 表从本机 `GraphOTTER-main` 复制补齐 |
| AIT-QA | `dataset/AIT-QA/aitqa_clean_questions.json` | 387 题 | 表格内嵌在问题 JSON 中，可直接运行 |

embedder 使用本机权重：
`/Users/mibo/paper/code/baseline_data/embedders/gte-base`（gte-base，210MB）。
对应的 HuggingFace 缓存为 `thenlper/gte-base`。

## 3. LLM backbone

论文使用 Qwen2-72B-Instruct（本地 vLLM）与 Gemini 1.5。本机没有 GPU/vLLM，
因此改走 **OpenAI 兼容 API**：

- 默认：SiliconFlow `Pro/deepseek-ai/DeepSeek-V3`
- 可切换：Bailian `qwen3.7-plus` 等，只要设置对应的 `GRAPHOTTER_MODEL` /
  `GRAPHOTTER_BASE_URL` / `GRAPHOTTER_API_KEY`。

凭据复用 `/Users/mibo/paper/code/ASTRA/.env`（脚本会 source，但不会打印）。
**这不是论文原始 backbone，数值不应与论文表格直接对齐**；复现重点是推理链路正确性。

## 4. 运行

冒烟测试（各 1 题）：

```bash
bash run_hitab.sh 0 1
bash run_aitqa.sh 0 1
```

小规模 / 全量：

```bash
bash run_hitab.sh 0 20        # HiTab 前 20 题
bash run_aitqa.sh 0 100       # AIT-QA 前 100 题
# HiTab 240 题等距子集（沿用 ASTRA 的 4 段：0-60/450-510/900-960/1350-1410 映射到 1584）
bash run_hitab.sh 0 240 subsets/hitab240_full_idx.json
bash run_hitab.sh 0 1584      # HiTab 全量
bash run_aitqa.sh 0 387       # AIT-QA 全量
```

日志与结果分别写到 `results/*.log`（当前 `main.py` 只打印 EM，不落 JSON）。

## 5. 为跑通而修改的官方代码

官方仓库存在几处会直接阻塞运行的缺陷，已做最小修复：

1. `main.py`：`--start/--end` 原为 `required=True`，但官方 `start.sh` 并未传，
   导致 argparse 直接退出。改为可选（默认全量）；`--debug` 改为 `store_true`。
2. `configs.py`：`iterative_reasoning.py` 引用了 `repeated_action_attention` 模板，
   但 `PROMPT_TEMPLATE` 里没有该键，触发重复动作时会 `KeyError`。已补模板。
3. `Generator/parse_output.py` + `iterative_reasoning.py`：最终答案 prompt 同时要求
   编号散文和 JSON，官方只按 JSON 解析；模型按散文作答时会误判失败。增加
   `parse_final_answer`（JSON 优先，散文兜底）。
4. `Generator/openai_api.py`：增加请求超时、指数退避重试，修复重试耗尽后 `resp`
   未定义的问题；对 Qwen3 显式关闭 thinking 模式。
5. `Generator/Gemini_model.py`：原先读取不存在的 `args.reasoning_model_path`，
   改为 `args.model`（仅 Gemini 路径）。
6. `GraphRetriever/dense_retriever.py`：macOS 上 `faiss-cpu` 与 torch 2.2.2 的
   OpenMP 运行时冲突，先 `import faiss` 再加载 SentenceTransformer 会直接
   `Segmentation fault: 11`。改为先加载模型、再延迟 `import faiss`。
7. `main.py`：增加逐题 try/except（单题失败重试 3 次后跳过，不中断整轮）、
   `--output_path` 结果落盘和 `--index_file` 子集选择；结果里的 `index` 使用
   原始数据集全局题号，分段跑完后可按题号合并，便于全量复现与统计。
8. `requirements-lock.txt` 增加 `safetensors==0.4.5`：本机 gte-base 只有
   `model.safetensors`，缺该包时 transformers 报 “no file named pytorch_model.bin”。

## 6. 冒烟测试结果（2026-11-03）

| 数据集 | 范围 | backbone | answered | EM |
|---|---|---|---|---|
| AIT-QA | [0,1) | DeepSeek-V3 | 1/1 | 1.0 |
| HiTab | [0,1) | DeepSeek-V3 | 1/1 | 1.0 |
| AIT-QA | [0,20) | DeepSeek-V3 | 20/20 | 1.0 |
| HiTab | [0,20) | DeepSeek-V3 | 20/20 | 0.85 |
| AIT-QA | [0,100) | DeepSeek-V3 | 100/100 | 0.97 |
| HiTab | 240 题等距子集 | DeepSeek-V3 | 240/240 | 0.8833 |
| AIT-QA | 全量 387 | DeepSeek-V3 | 387/387 | 0.8811 |

AIT-QA 全量分题型：KPI-driven 90/99 = 0.9091，Table-driven 251/288 = 0.8715，
共 46 题错误、0 题失败。全量结果在 `results/aitqa_full_0-387.json`。

两条链路均已完成：LLM 选单元格 → gte-base/BM25 检索 → 子图构建 →
Thought/Action/Observation → 最终答案解析。逐题结果落在
`results/<dataset>_<tag>.json`，日志为同名 `.log`。

对照：同样的 HiTab 240 子集在 `GraphOTTER-main`（旧复现）里 EM=0.875，
本次 0.8833，量级一致；差异来自模型随机性与本次新增的散文答案解析兜底。
240 题运行中只有 1 次 action JSON 解析报错，重试后成功，最终 0 道失败题。

## 7. 已知限制

- `main.py` 默认只计算 **EM**；`LLM_eval` 在官方代码中被注释掉。
- 官方 HiTab 数据预处理会去掉表头标注；本复现直接使用仓库附带/补齐的 raw 表。
- 未启用并发。串行实测约 23–28 秒/题；AIT-QA 387 题已完整跑完，HiTab 1584 题
  全量尚未运行（240 题子集已完成）。按当前速度全量 HiTab 约需 12 小时，需要时
  可在后续加入线程池。
