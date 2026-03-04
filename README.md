# WinterWen-national-security-methods

本仓库为硕士学位论文《国家安全学研究方法的演进、评估与规范》的线上材料库，用于集中存放论文实证部分的过程性文件、数据表、变量字典、SCALE编码相关材料，以及贝叶斯网络（Bayesian Network, BN）分析脚本与稳健性输出，便于同行在不接触原始全文材料的前提下，对论文的研究设计与操作口径进行查验、复核与检验性阅读。:contentReference[oaicite:1]{index=1}

> 说明：本仓库不等同于“可一键复现”的软件工程项目；其定位是支持论文审阅时对关键口径（变量定义、离散化规则、编码结果、模型设定与稳健性证据）的核对。:contentReference[oaicite:2]{index=2}

---

## 目录与文件说明（按论文查验需求组织）

### 1. 过程说明
- `过程说明/研究流程阅读指南.docx`：对论文实证主线SCALE编码 → 封版数据 → BN（Bayesian Network）建模 → 稳健性/稳定性证据）的操作步骤与关键参数进行汇总说明。:contentReference[oaicite:3]{index=3}
- `过程说明/SCALE简化版操作手册.docx`：SCALE编码流程的操作规程与记录要点（包括编码—讨论—代码簿演化—一致性度量等）。:contentReference[oaicite:4]{index=4}

### 2. 数据预处理与前期准备
- `国家安全学相关期刊论文数据预处理.xlsx`：样本预处理与前期整理的记录性表格（用于核对样本筛选、字段构造与输入表形成过程）。:contentReference[oaicite:5]{index=5}
- `数据预处理和前期准备部分/`：与预处理步骤相关的补充材料（如有）。:contentReference[oaicite:6]{index=6}

### 3. SCALE相关材料（编码、校准、讨论、结果转化）
以下目录用于支撑对“SCALE如何从文本证据生成结构化变量”的核对：
- `SCALE_Agent/`：与多代理（agents）编码相关的提示词、设定与过程材料（如有）。:contentReference[oaicite:7]{index=7}
- `SCALE_校准轮及结果汇总后的分析/`：校准阶段材料与汇总分析（如有）。:contentReference[oaicite:8]{index=8}
- `SCALE_讨论轮/`：冲突池讨论与仲裁相关材料（如有）。:contentReference[oaicite:9]{index=9}
- `SCALE_结果材料/`、`SCALE_结果转化/`：编码结果与从结果到封版/入模表的转换材料（如有）。:contentReference[oaicite:10]{index=10}

### 4. BN（Bayesian Network）建模与核心输入（建议优先查验）
- `BN代码/`：BN（Bayesian Network）分析脚本与关键规范文件：:contentReference[oaicite:11]{index=11}  
  - `bn_pipeline_v2.py`：主要建模脚本（结构学习、参数估计、稳健性/稳定性输出等）。  
  - `inference_utils.py`：推断与工具函数。  
  - `bn_nodes_spec_v2.md`：BN 节点/变量设定说明。  
  - `variable_dictionary_v2.tsv`：变量字典（字段含义与取值说明）。  
  - `codebook_v2_elements.tsv`：要素层代码簿（与编码口径相关）。  
  - `scale_labels_bn_ready.csv`：BN 直接使用的数据表（行=论文样本，列=离散化变量）。  
  - `results_run_20251009_164600/`：一次运行的输出目录快照（用于核对输出表/图与论文一致性）。  

- `BN准备文件/`：BN 入模前的准备性文件与仲裁补丁：:contentReference[oaicite:12]{index=12}  
  - `BN-就绪总表（已合并仲裁补丁，离散化完成）.csv`：入模前整理完成的汇总表（含离散化）。  
  - `adjudicated_patch_top100.csv`、`adjudicated_gold_top100.csv`、`adjudication_summary_sources.csv`：仲裁补丁、金标/抽检与来源汇总（用于核对仲裁与质量控制）。  
  - `变量字典（v2，字段取值释义齐备）.tsv`：变量字典 v2。  
  - `要素层完整代码簿 v2（非速查版；含“包含排除边界锚点映射”）.tsv`：更完整的代码簿与边界说明。  

### 5. 论文附录与稳健性证据
- `附录部分/`：论文附录表格与稳健性/稳定性证据文件集合：:contentReference[oaicite:13]{index=13}  
  - `A.xlsx`、`B.xlsx`、`BNA.xlsx`、`M1.xlsx`：附录表格（变量字典/统计表/BN附录/映射表等）。  
  - `bootstrap_edges_A.csv`、`bootstrap_edges_B.csv`：自举（bootstrap）边频率结果。  
  - `adjacency_stable_A.csv`、`adjacency_stable_B.csv`：稳定邻接矩阵输出。  
  - `《26→9 映射表.tsv》（映射版本号与双向追溯）.tsv`：方法门类映射与追溯信息。  
  - `最终编码表.csv`：编码结果的汇总输出（用于与论文表格/统计复核）。  
  - `附录A 变量字典 代码本.docx`、`国家安全学研究方法基础界定规则与分类列表.docx`：附录说明与规则文本。  

---

## 查验要点（面向论文审阅的“快速核对”）

1. **变量与取值口径**：优先核对 `BN代码/variable_dictionary_v2.tsv` 与 `BN代码/bn_nodes_spec_v2.md` 的定义是否与论文正文/附录一致。:contentReference[oaicite:14]{index=14}  
2. **入模数据**：优先核对 `BN代码/scale_labels_bn_ready.csv` 与 `BN准备文件/BN-就绪总表（已合并仲裁补丁，离散化完成）.csv`，确认样本行数、字段集合与离散化方式一致。:contentReference[oaicite:15]{index=15}  
3. **模型实现与输出**：核对 `BN代码/bn_pipeline_v2.py` 与 `BN代码/results_run_20251009_164600/` 的输出文件是否对应论文中使用的表格/图示与稳健性证据。:contentReference[oaicite:16]{index=16}  
4. **稳健性/稳定性证据**：核对 `附录部分/bootstrap_edges_*.csv` 与 `附录部分/adjacency_stable_*.csv`，并与论文对“稳定边/稳定骨架/自举频率阈值”等口径的描述对应。:contentReference[oaicite:17]{index=17}  

---

## 合规与边界说明

- 本仓库不包含原始论文全文（如CNKI下载PDF/全文文本），仅提供编码结果、变量字典、代码簿、脚本与输出证据，以避免版权与敏感材料外泄风险。  
- 若仓库中出现任何可识别的原文长段落或版权受限内容，应以“证据摘录最小化”原则处理（例如仅保留方法识别所需的短句锚点），并以 paper_id/题录信息支持回溯。  

---

## 许可（License）
本仓库使用 MIT License（详见 `LICENSE`）。:contentReference[oaicite:18]{index=18}
