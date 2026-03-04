# -*- coding: utf-8 -*-
"""
Bayesian Network Pipeline v2.2 (全量替换版)
- 修复：HC 起始先验只作 start_dag，不再硬白名单；避免空图/全零稳定矩阵。
- 约束：YearBin, topic 设为外生（禁止任何 *→YearBin / *→topic）。
- 健壮：自动剔除常数列学习结构，导出时补回孤点；Matplotlib 使用 'Agg'，避免 Tk 异常。
- 产出：HC/PC 结构图、仅稳定边骨架图、交叉验证、Bootstrap 稳定性、代码本、声明文本、summary.json。
"""

import os, json, math, itertools, warnings, random, datetime as dt
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# 后端固定为 Agg，避免桌面环境下 Tk 错误
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.model_selection import KFold

from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, PC
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(suppress=True)

# ------------------------- 全局参数 -------------------------
RANDOM_SEED = 20251008
N_FOLDS     = 5
ESS         = 1.5             # Dirichlet 等效样本量
N_BOOT      = 200             # Bootstrap 次数
STABLE_TH   = 0.70            # 稳定边阈值（论文声明用）
DATA_FILE   = "./scale_labels_bn_ready.csv"

# ------------------------- 工具函数 -------------------------
def nowstr():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def log(*args):
    print(*args, flush=True)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def discretize_topic(s):
    """把 topic 离散化为字符串类别（若已离散则原样返回）"""
    if pd.api.types.is_numeric_dtype(s):
        # 保留原值但转为字符串，以避免被当作连续分箱
        return s.fillna(-1).astype(int).astype(str)
    return s.fillna("NA").astype(str)

def make_year_bin(year_series, bins=None):
    """Year → YearBin 离散化"""
    y = pd.to_numeric(year_series, errors="coerce")
    y = y.fillna(y.median())
    # 默认以四分位作分箱（更稳妥）
    if bins is None:
        qs = np.quantile(y, [0, 0.25, 0.5, 0.75, 1.0])
        bins = sorted(set(int(v) for v in qs))
        if len(bins) < 3:  # 兜底
            bins = [int(y.min()), int(np.median(y)), int(y.max())]
    # 去重并确保严格递增
    bins = sorted(set(bins))
    if len(bins) < 3:
        # 极端情况下使用三等分
        bins = [int(y.min()), int((y.min()+y.max())/2), int(y.max())]
    cats = pd.cut(y, bins=len(bins)-1, labels=[f"Y{i+1}" for i in range(len(bins)-1)], include_lowest=True)
    return cats.astype(str)

def fill_method_normativity(df):
    """若缺失 MethodNormativity：由 MethodTransparency 粗略映射（H/M/L→3/2/1；未知→0）"""
    if "MethodNormativity" not in df.columns:
        mp = {"H": 3, "M": 2, "L": 1}
        src = df.get("MethodTransparency")
        if src is None:
            df["MethodNormativity"] = 0
        else:
            df["MethodNormativity"] = src.map(mp).fillna(0).astype(int)
        log("[FIX] 缺失列 MethodNormativity：已根据 MethodTransparency 生成近似等级（H/M/L→3/2/1；未知→0）。")
    return df

def select_scheme(df):
    """两套方案的字段选择"""
    cols_A = ["YearBin", "topic",
              "E_EXP","E_STAT","E_SURVEY","E_CASE","E_ETHNO","E_TEXT","E_MODEL","E_POLICYEVAL","E_MIXED","E_NORM",
              "PolicySalience","PoliticalSensitivity","SecrecyConstraint","DataAccess","MethodTransparency"]

    cols_B = ["YearBin", "topic",
              "E_EXP","E_STAT","E_SURVEY","E_CASE","E_ETHNO","E_TEXT","E_MODEL","E_POLICYEVAL","E_MIXED","E_NORM",
              "MethodFamily","MethodNormativity",
              "PolicySalience","PoliticalSensitivity","SecrecyConstraint","DataAccess","MethodTransparency"]

    dataA = df[[c for c in cols_A if c in df.columns]].copy()
    dataB = df[[c for c in cols_B if c in df.columns]].copy()

    log("[SCHEME A] 选取列:", list(dataA.columns))
    log("[SCHEME B] 选取列:", list(dataB.columns))
    return dataA, dataB

def white_black_lists(cols, scheme, enable_h3=True):
    """构建起始先验与黑名单（数据驱动为主）。
       - WL 仅作为 start_dag 起点，不作硬白名单。
       - 外生性：YearBin、topic 禁止任何入边（*→YearBin/topic）。
       - B 方案对 MethodFamily 做方向限制（与前版脚本一致的主旨）。
    """
    cols = set(cols)
    wl = []
    bl = set()

    # 外生性：禁止任何 *→YearBin, *→topic
    for c in cols:
        if c != "YearBin":
            bl.add((c, "YearBin"))
        if c != "topic":
            bl.add((c, "topic"))

    # 少量温和先验（仅作为起始边）
    if enable_h3:
        if "PoliticalSensitivity" in cols and "SecrecyConstraint" in cols:
            wl.append(("PoliticalSensitivity", "SecrecyConstraint"))
        if "SecrecyConstraint" in cols and "MethodTransparency" in cols:
            wl.append(("SecrecyConstraint", "MethodTransparency"))
        if "topic" in cols and "MethodFamily" in cols:
            wl.append(("topic","MethodFamily"))
        if "PolicySalience" in cols and "MethodFamily" in cols:
            wl.append(("PolicySalience","MethodFamily"))

    # 方案 B：限制 MethodFamily 的不合理外出方向（延续前版思路）
    if scheme == "B" and "MethodFamily" in cols:
        for tgt in ["PoliticalSensitivity","PolicySalience","SecrecyConstraint","DataAccess"]:
            if tgt in cols:
                bl.add(("MethodFamily", tgt))

    return wl, list(bl)

def ensure_nodes(model_or_edges, cols):
    """
    接受:
      - pgmpy.models.BayesianNetwork
      - networkx.DiGraph
      - list/tuple 的 (u, v) 边列表
      - networkx 的 EdgeView / OutEdgeView / InEdgeView 等“边视图”
      - 其它任意可迭代且元素为二元组的对象
    返回: 含有 cols 中全部节点的 BayesianNetwork（缺的补成孤点）
    """
    from pgmpy.models import BayesianNetwork
    import networkx as nx
    from collections.abc import Iterable

    # 1) 标准化为 BayesianNetwork
    if isinstance(model_or_edges, BayesianNetwork):
        bn = BayesianNetwork(list(model_or_edges.edges()))
        bn.add_nodes_from(list(model_or_edges.nodes()))
    elif isinstance(model_or_edges, nx.DiGraph):
        bn = BayesianNetwork(list(model_or_edges.edges()))
        bn.add_nodes_from(list(model_or_edges.nodes()))
    else:
        # 尝试把“可迭代的边视图/列表”等统一转成 list[(u,v)]
        if isinstance(model_or_edges, Iterable) and not isinstance(model_or_edges, (str, bytes)):
            edges = []
            for e in model_or_edges:
                # 允许 e 为 networkx 的 EdgeDataView 三元组，或普通二元组
                if isinstance(e, (list, tuple)):
                    if len(e) >= 2:
                        edges.append((e[0], e[1]))
                    else:
                        continue
                else:
                    # 遇到非二元元素就跳过（更稳妥）
                    continue
            bn = BayesianNetwork(edges)
        else:
            raise TypeError(f"Unsupported type for ensure_nodes: {type(model_or_edges)}")

    # 2) 把缺失节点补上（作为孤点）
    current_nodes = set(bn.nodes())
    missing = [c for c in cols if c not in current_nodes]
    if missing:
        bn.add_nodes_from(missing)

    return bn

def draw_dag(model, path_png, title=None, stable_edges=None):
    """画 DAG 并保存，尽量使用层次布局；失败则回退 spring_layout。"""
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())
    plt.figure(figsize=(10,8))
    try:
        # 优先 dot 布局（若安装了 pygraphviz 或 pydot）
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        try:
            pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="#E0ECF4", edgecolors="#4E79A7")
    nx.draw_networkx_labels(G, pos, font_size=8)
    edgelist = list(G.edges())
    colors = []
    for e in edgelist:
        if stable_edges and e in stable_edges:
            colors.append("#D62728")  # 稳定边用红色
        else:
            colors.append("#7F7F7F")
    nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color=colors, arrows=True, width=1.5, arrowsize=12)
    if title:
        plt.title(title, fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()

def save_adj_heatmap(nodes, edge_freq, out_png, title):
    """稳定性邻接矩阵热力图"""
    idx = {n:i for i,n in enumerate(nodes)}
    M = np.zeros((len(nodes), len(nodes)), dtype=float)
    for (u,v),f in edge_freq.items():
        if u in idx and v in idx:
            M[idx[u], idx[v]] = f
    plt.figure(figsize=(10,8))
    plt.imshow(M, interpolation="nearest")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(nodes)), nodes, rotation=90, fontsize=7)
    plt.yticks(range(len(nodes)), nodes, fontsize=7)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def export_codebook(df, out_csv):
    """导出代码本（基础信息：变量名、数据类型、唯一值个数、示例取值）"""
    rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        nunq = s.nunique(dropna=True)
        sample_vals = list(pd.Series(s.unique()).dropna().astype(str).head(6))
        rows.append({
            "variable": col,
            "dtype": dtype,
            "n_unique": int(nunq),
            "examples": "; ".join(sample_vals)
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

# ------------------------- 结构学习 -------------------------
def learn_hc_bic(df: pd.DataFrame, scheme: str, enable_h3_whitelist: bool = True):
    # 1) 剔除常数列（唯一取值<=1）
    nunq = df.nunique(dropna=False)
    drop_cols = [c for c,k in nunq.items() if int(k) <= 1]
    data = df.drop(columns=drop_cols, errors="ignore").copy()

    scorer = BicScore(data)
    hc = HillClimbSearch(data)
    cols = list(data.columns)
    WL, BL = white_black_lists(cols, scheme, enable_h3_whitelist)

    # 2) WL 仅作为起始先验（start_dag）
    prior_edges = []
    for (u, v) in WL or []:
        if (u in cols) and (v in cols):
            prior_edges.append((u, v))
    prior_edges = list(set(prior_edges))

    # 关键修补：start_dag 必须含有与数据集一致的节点全集
    if prior_edges:
        start_dag = BayesianNetwork()
        start_dag.add_nodes_from(cols)      # ← 加入所有列作为节点（即使是孤点）
        start_dag.add_edges_from(prior_edges)
    else:
        start_dag = None

    log(f"[HC] scheme={scheme} | start_edges={prior_edges} | black_list={BL} | drop_const={drop_cols}")

    try:
        est = hc.estimate(
            scoring_method=scorer,
            white_list=None,
            black_list=BL if BL else None,
            start_dag=start_dag,
            max_indegree=None,
            max_iter=200,
            show_progress=False
        )
    except TypeError:
        # 某些版本参数不同
        est = hc.estimate(scoring_method=scorer)

    bn = ensure_nodes(est, list(data.columns))
    log(f"[HC] scheme={scheme} | learned_edges={len(bn.edges())}")
    return bn

def learn_pc(df: pd.DataFrame, scheme: str):
    data = df.copy()
    cols = list(data.columns)
    WL, BL = white_black_lists(cols, scheme, enable_h3=True)
    # PC 不支持白/黑名单，使用独立性检验学习后再过滤方向：这里仅学习无向骨架，然后用简单方向化
    pc = PC(data)
    try:
        skel, sep = pc.build_skeleton(significance_level=0.01, max_cond_vars=5, variant="stable")
    except TypeError:
        skel, sep = pc.build_skeleton(significance_level=0.01)
    undirected = nx.Graph()
    undirected.add_nodes_from(cols)
    undirected.add_edges_from(skel.edges())

    # 简易方向化（不与黑名单冲突就按 WL 方向化；其余保持无向→双向占位，后续作图时仍显示）
    directed = nx.DiGraph()
    directed.add_nodes_from(cols)

    # 先按 WL 定向
    for (u,v) in WL:
        if undirected.has_edge(u,v) and (u,v) not in BL:
            directed.add_edge(u,v)

    # 对剩余无向边，任取固定方向（按节点名排序）但若命中黑名单则反向；若仍冲突则放弃
    for (u,v) in undirected.edges():
        if directed.has_edge(u,v) or directed.has_edge(v,u):
            continue
        cand = (min(u,v), max(u,v))
        a,b = cand
        if (a,b) not in BL:
            directed.add_edge(a,b)
        elif (b,a) not in BL:
            directed.add_edge(b,a)
        # 若两向都在 BL，放弃

    bn = ensure_nodes(directed.edges(), cols)
    log(f"[PC] scheme={scheme} | learned_edges={len(bn.edges())}")
    return bn

# ------------------------- 参数估计 -------------------------
def estimate_params(bn: BayesianNetwork, data: pd.DataFrame):
    model = BayesianNetwork(bn.edges())
    model.add_nodes_from(bn.nodes())
    model.fit(data, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=ESS)
    return model

# ------------------------- 交叉验证 -------------------------
def cross_validate(df: pd.DataFrame, scheme: str, out_json: str):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = []
    i = 0
    for tr_idx, te_idx in kf.split(df):
        i += 1
        train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        model = learn_hc_bic(train, scheme)
        # 用测试集评分（BIC/K2）
        bic = BicScore(test).score(model)
        k2  = K2Score(test).score(model)
        log(f"[CV] scheme={scheme} fold={i} | BIC={bic:.3f} | K2={k2:.3f}")
        folds.append({"fold": i, "BIC": float(bic), "K2": float(k2)})
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(folds, f, ensure_ascii=False, indent=2)

# ------------------------- 自举稳定性 -------------------------
def bootstrap_edges(df: pd.DataFrame, scheme: str, n_boot=N_BOOT, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    edge_counter = Counter()
    cols = list(df.columns)
    for b in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        samp = df.iloc[idx].copy()
        bn = learn_hc_bic(samp, scheme)
        edge_counter.update(list(bn.edges()))
    # 频率
    edge_freq = {e: edge_counter[e] / n_boot for e in edge_counter}
    return edge_freq

# ------------------------- 导出助手 -------------------------
def export_edge_tables(edge_freq, nodes, out_csv, out_png, title):
    # CSV
    rows = [{"u": u, "v": v, "freq": f} for (u,v),f in sorted(edge_freq.items(), key=lambda x: (-x[1], x[0]))]
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    # 热力图
    save_adj_heatmap(nodes, edge_freq, out_png, title)

def edges_to_adj(model):
    nodes = list(model.nodes())
    idx = {n:i for i,n in enumerate(nodes)}
    M = np.zeros((len(nodes), len(nodes)), dtype=int)
    for u,v in model.edges():
        M[idx[u], idx[v]] = 1
    df = pd.DataFrame(M, index=nodes, columns=nodes)
    return df

def stable_skeleton(edge_freq, th=STABLE_TH):
    return [e for e,f in edge_freq.items() if f >= th]

# ------------------------- 主流程 -------------------------
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    log("=== BN 主线脚本启动（v2.2 修订） ===")
    log("随机种子：", RANDOM_SEED)

    run_dir = f"./results_run_{nowstr()}"
    ensure_dir(run_dir)
    log("运行目录：", run_dir)

    # 读数据
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    log(f"[DATA] 读取 {DATA_FILE} | 形状={df.shape} | 列={list(df.columns)}")

    # YearBin / topic 处理
    if "year" in df.columns and "YearBin" not in df.columns:
        df["YearBin"] = make_year_bin(df["year"])
        log("[FIX] 年份已离散化为 YearBin。")
    if "topic" in df.columns:
        df["topic"] = discretize_topic(df["topic"])
        nunq = df["topic"][df["topic"]!="0"].nunique()
        log(f"[FIX] 主题列已离散化：unique_nonzero={nunq}")

    # MethodNormativity 补列
    df = fill_method_normativity(df)

    # 方案选择
    dataA, dataB = select_scheme(df)

    # ---------- 结构学习 ----------
    log("[A] 结构学习（HC+BIC）...")
    modelA_hc = learn_hc_bic(dataA, scheme="A")

    log("[B] 结构学习（HC+BIC）...")
    modelB_hc = learn_hc_bic(dataB, scheme="B")

    log("[A] 结构学习（PC，辅）...")
    modelA_pc = learn_pc(dataA, scheme="A")

    log("[B] 结构学习（PC，辅）...")
    modelB_pc = learn_pc(dataB, scheme="B")

    # ---------- 参数估计 ----------
    log(f"[A] 参数估计（Dirichlet, ESS={ESS}）")
    modelA_full = estimate_params(modelA_hc, dataA)
    log(f"[B] 参数估计（Dirichlet, ESS={ESS}）")
    modelB_full = estimate_params(modelB_hc, dataB)

    # ---------- 全样本评分 ----------
    scoreA_bic = BicScore(dataA).score(modelA_hc)
    scoreA_k2  = K2Score(dataA).score(modelA_hc)
    log(f"[A] Full-sample | BIC={scoreA_bic:.3f} | K2={scoreA_k2:.3f}")

    scoreB_bic = BicScore(dataB).score(modelB_hc)
    scoreB_k2  = K2Score(dataB).score(modelB_hc)
    log(f"[B] Full-sample | BIC={scoreB_bic:.3f} | K2={scoreB_k2:.3f}")

    # ---------- 交叉验证 ----------
    cvA_json = os.path.join(run_dir, "cv_A.json")
    cvB_json = os.path.join(run_dir, "cv_B.json")
    cross_validate(dataA, "A", cvA_json)
    cross_validate(dataB, "B", cvB_json)

    # ---------- 自举稳定性 ----------
    log(f"[BOOT] Scheme A | n_boot={N_BOOT}")
    edge_freq_A = bootstrap_edges(dataA, "A", n_boot=N_BOOT, seed=RANDOM_SEED)
    log(f"[BOOT] Scheme B | n_boot={N_BOOT}")
    edge_freq_B = bootstrap_edges(dataB, "B", n_boot=N_BOOT, seed=RANDOM_SEED)

    # 导出稳定性
    nodesA = list(modelA_hc.nodes())
    nodesB = list(modelB_hc.nodes())

    export_edge_tables(edge_freq_A, nodesA,
                       os.path.join(run_dir, "bootstrap_edges_A.csv"),
                       os.path.join(run_dir, "bootstrap_adjmat_A.png"),
                       f"Bootstrap Edge Frequency (A)")

    export_edge_tables(edge_freq_B, nodesB,
                       os.path.join(run_dir, "bootstrap_edges_B.csv"),
                       os.path.join(run_dir, "bootstrap_adjmat_B.png"),
                       f"Bootstrap Edge Frequency (B)")

    # 稳定骨架
    stableA = stable_skeleton(edge_freq_A, STABLE_TH)
    stableB = stable_skeleton(edge_freq_B, STABLE_TH)

    # ---------- 作图 ----------
    # HC/PC 全图
    draw_dag(modelA_hc, os.path.join(run_dir, "dag_hc_A.png"), title="HC (Scheme A)")
    draw_dag(modelB_hc, os.path.join(run_dir, "dag_hc_B.png"), title="HC (Scheme B)")
    draw_dag(modelA_pc, os.path.join(run_dir, "dag_pc_A.png"), title="PC (Scheme A)")
    draw_dag(modelB_pc, os.path.join(run_dir, "dag_pc_B.png"), title="PC (Scheme B)")

    # 仅稳定边骨架图（以 HC 节点集为底座）
    modelA_stable = ensure_nodes(stableA, nodesA)
    modelB_stable = ensure_nodes(stableB, nodesB)
    draw_dag(modelA_stable, os.path.join(run_dir, "dag_stable_A.png"),
             title=f"Stable Skeleton (A) freq≥{STABLE_TH}", stable_edges=set(stableA))
    draw_dag(modelB_stable, os.path.join(run_dir, "dag_stable_B.png"),
             title=f"Stable Skeleton (B) freq≥{STABLE_TH}", stable_edges=set(stableB))

    # 邻接矩阵导出
    edges_to_adj(modelA_hc).to_csv(os.path.join(run_dir, "adjacency_hc_A.csv"), encoding="utf-8-sig")
    edges_to_adj(modelB_hc).to_csv(os.path.join(run_dir, "adjacency_hc_B.csv"), encoding="utf-8-sig")
    edges_to_adj(modelA_stable).to_csv(os.path.join(run_dir, "adjacency_stable_A.csv"), encoding="utf-8-sig")
    edges_to_adj(modelB_stable).to_csv(os.path.join(run_dir, "adjacency_stable_B.csv"), encoding="utf-8-sig")

    # 代码本
    export_codebook(df, os.path.join(run_dir, "codebook.csv"))

    # 声明文本（直接可粘贴到论文）
    with open(os.path.join(run_dir, "readme.txt"), "w", encoding="utf-8") as f:
        f.write("先验敏感性声明：\n")
        f.write("  结构学习阶段仅将领域先验作为起始边（start_dag），而非硬性白名单；因此结果主要由数据驱动，避免先验对结构的过度支配。\n\n")
        f.write("稳定边阈值声明：\n")
        f.write(f"  使用自举法（n={N_BOOT}）评估结构稳健性，本研究以出现频率≥{STABLE_TH:.2f} 判定为稳定边，论文只对稳定骨架进行解释。\n\n")
        f.write("外生性约束：\n")
        f.write("  YearBin 与 topic 被视为外生变量，模型禁止任何指向它们的入边（*→YearBin / *→topic）。\n\n")

    # 总括 summary
    def summer(edge_freq):
        vals = sorted(edge_freq.values(), reverse=True)
        if not vals:
            return {"n_edges": 0}
        return {
            "n_edges": len(vals),
            "freq_mean": float(np.mean(vals)),
            "freq_median": float(np.median(vals)),
            "n_ge_th": int(np.sum(np.array(vals) >= STABLE_TH)),
            "top5": sorted(edge_freq.items(), key=lambda x: -x[1])[:5]
        }

    summary = {
        "seed": RANDOM_SEED,
        "run_dir": run_dir,
        "cv_files": {"A": os.path.basename(cvA_json), "B": os.path.basename(cvB_json)},
        "scores_full": {
            "A": {"BIC": float(scoreA_bic), "K2": float(scoreA_k2)},
            "B": {"BIC": float(scoreB_bic), "K2": float(scoreB_k2)}
        },
        "bootstrap": {
            "n_boot": N_BOOT,
            "stable_threshold": STABLE_TH,
            "A": summer(edge_freq_A),
            "B": summer(edge_freq_B)
        },
        "notes": [
            "使用 HillClimb+BIC 为主、PC 为辅；Dirichlet 平滑 ESS=1.5。",
            "YearBin 与 topic 外生；MethodTransparency 若为常数将自动在学习阶段被忽略，但在作图时作为孤点补回。"
        ]
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("=== 运行完成 ===")

# ------------------------- 入口 -------------------------
if __name__ == "__main__":
    main()
