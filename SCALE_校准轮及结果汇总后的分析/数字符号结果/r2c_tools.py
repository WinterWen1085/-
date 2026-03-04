# r2c_tools.py  —— R2-C 修复工具包：数值化标准化 + 冲突矩阵 + 快速α
# 用法：
# 1) 数值化：python r2c_tools.py sanitize --in agent_A.csv --out scale_labels_calib_agentA.tsv
# 2) 冲突矩阵：python r2c_tools.py conflicts --in agent_A.tsv agent_B.tsv agent_C.tsv agent_D.tsv --out conflicts_matrix.tsv
# 3) 快速α：python r2c_tools.py alpha --in agent_A.tsv agent_B.tsv agent_C.tsv agent_D.tsv --out alpha_report.tsv
# 映射表（可选，同目录）：methodfamily_map.tsv / methodology_map.tsv / normativity_map.tsv：两列 label<TAB>code

import argparse, sys, os, csv, math
from collections import defaultdict, Counter
import pandas as pd

VAR_ORDER = ["paper_id","MethodTag_codes","MethodFamily_codes","Methodology_codes","MethodTransparency_codes","MethodNormativity_codes","coder_notes_rule_code"]

MAP_MethodTag = {"10":10,"20":20,"30":30,"40":40,
    "Normative":10,"规范":10,"法理":10,"规范法学":10,
    "Empirical":20,"实证":20,"经验研究":20,
    "Technical":30,"技术":30,"建模":30,"模型":30,"仿真":30,
    "Philosophical":40,"思辨":40,"哲学":40,"话语":40}

MAP_MethodTransparency = {"1":1,"2":2,"3":3,"L":1,"Low":1,"低":1,"M":2,"Medium":2,"中":2,"H":3,"High":3,"高":3}

MAP_FILES = {"MethodFamily_codes":"methodfamily_map.tsv","Methodology_codes":"methodology_map.tsv","MethodNormativity_codes":"normativity_map.tsv"}

FALLBACK_MethodFamily = {"102":102,"比较研究":102,"199":199,"规范/思辨（不充分）":199}
FALLBACK_Methodology = {"1":1,"规范":1,"normative":1,"2":2,"定量":2,"quantitative":2,"3":3,"定性":3,"qualitative":3,"4":4,"混合":4,"mixed":4,"5":5,"技术":5,"technical":5}
FALLBACK_Normativity = {"1":1,"不足":1,"2":2,"一般":2,"3":3,"充分":3}

def load_mapfile(path):
    d={}
    if not os.path.exists(path): return d
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            parts=line.split("\t")
            if len(parts)<2: continue
            label,code=parts[0].strip(),parts[1].strip()
            if code.isdigit(): d[label]=int(code)
    return d

def coerce_int(x):
    try:
        if x is None: return None
        if isinstance(x,float) and math.isnan(x): return None
        sx=str(x).strip()
        if sx=="": return None
        return int(float(sx))
    except: return None

def apply_map(series, mapping, name):
    out=[]
    for v in series:
        if v is None or (isinstance(v,float) and math.isnan(v)): out.append(None); continue
        s=str(v).strip()
        if s in mapping: out.append(mapping[s]); continue
        if s.isdigit(): out.append(int(s)); continue
        out.append(None)
    return out

def sanitize_one(in_path, out_path=None, agent_name=None):
    df=pd.read_csv(in_path, sep=None, engine="python", dtype=str, keep_default_na=False)

    rename={}
    for c in df.columns:
        cs=c.strip()
        if cs.lower() in ["paper_id","id"]: rename[c]="paper_id"
        elif "MethodTag" in cs: rename[c]="MethodTag"
        elif "MethodFamily" in cs: rename[c]="MethodFamily"
        elif "Methodology" in cs: rename[c]="Methodology"
        elif "Transparency" in cs: rename[c]="MethodTransparency"
        elif "Normativity" in cs: rename[c]="MethodNormativity"
        elif "coder" in cs and "rule" in cs: rename[c]="coder_notes_rule_code"
        elif cs in ["方法标签","方法大类","方法学类型","方法透明度","规范度","备注规则码"]:
            zh_map={"方法标签":"MethodTag","方法大类":"MethodFamily","方法学类型":"Methodology","方法透明度":"MethodTransparency","规范度":"MethodNormativity","备注规则码":"coder_notes_rule_code"}
            rename[c]=zh_map[cs]
    df=df.rename(columns=rename)

    required=["paper_id","MethodTag","MethodFamily","Methodology","MethodTransparency","MethodNormativity","coder_notes_rule_code"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise SystemExit(f"[ERROR] 缺少必要列: {missing} in {in_path}")

    mf_map=load_mapfile(MAP_FILES["MethodFamily_codes"]) or FALLBACK_MethodFamily.copy()
    mo_map=load_mapfile(MAP_FILES["Methodology_codes"]) or FALLBACK_Methodology.copy()
    no_map=load_mapfile(MAP_FILES["MethodNormativity_codes"]) or FALLBACK_Normativity.copy()

    out=pd.DataFrame()
    out["paper_id"]=[coerce_int(x) for x in df["paper_id"]]
    out["MethodTag_codes"]=apply_map(df["MethodTag"], MAP_MethodTag, "MethodTag")
    out["MethodFamily_codes"]=apply_map(df["MethodFamily"], mf_map, "MethodFamily")
    out["Methodology_codes"]=apply_map(df["Methodology"], mo_map, "Methodology")
    out["MethodTransparency_codes"]=apply_map(df["MethodTransparency"], MAP_MethodTransparency, "MethodTransparency")
    out["MethodNormativity_codes"]=apply_map(df["MethodNormativity"], no_map, "MethodNormativity")

    def norm_rule(v):
        iv=coerce_int(v)
        if iv is None: return None
        if 901<=iv<=909: return iv
        return None
    out["coder_notes_rule_code"]=[norm_rule(v) for v in df["coder_notes_rule_code"]]

    violations={}
    for col in VAR_ORDER:
        nulls=sum([1 for x in out[col] if x is None])
        violations[col]={"nulls":nulls}
    domain_checks={
        "MethodTag_codes":lambda x: x in [10,20,30,40] if x is not None else True,
        "MethodTransparency_codes":lambda x: x in [1,2,3] if x is not None else True,
        "coder_notes_rule_code":lambda x: (x is None) or (901<=x<=909)
    }
    for col,fn in domain_checks.items():
        bad=sum([1 for x in out[col] if not fn(x)])
        violations[col]["domain_bad"]=bad

    out=out[VAR_ORDER]

    print(f"[OK] 读取 {in_path}: {len(out)} 行")
    for k,v in violations.items():
        print(f"  - {k}: 缺失 {v['nulls']}", end="")
        if "domain_bad" in v: print(f"; 越界 {v['domain_bad']}", end="")
        print()

    if out_path:
        out.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
        print(f"[OK] 写出 {out_path}")
    else:
        print(out.head().to_string(index=False))

def stack_agents(paths):
    frames=[]
    for p in paths:
        df=pd.read_csv(p, sep=None, engine="python", dtype=str, keep_default_na=False)
        cols=set([c.strip() for c in df.columns])
        if "MethodTag_codes" not in cols:
            tmp_out=sanitize_in_memory(df)
        else:
            tmp_out=df.copy()
        agent=os.path.basename(p).split(".")[0]
        tmp_out["agent"]=agent
        for c in VAR_ORDER:
            tmp_out[c]=tmp_out[c].apply(coerce_int)
        frames.append(tmp_out[VAR_ORDER+["agent"]])
    long=pd.concat(frames, ignore_index=True)
    return long

def sanitize_in_memory(df):
    mf_map=load_mapfile(MAP_FILES["MethodFamily_codes"]) or FALLBACK_MethodFamily
    mo_map=load_mapfile(MAP_FILES["Methodology_codes"]) or FALLBACK_Methodology
    no_map=load_mapfile(MAP_FILES["MethodNormativity_codes"]) or FALLBACK_Normativity
    out=pd.DataFrame()
    for k in ["paper_id","MethodTag","MethodFamily","Methodology","MethodTransparency","MethodNormativity","coder_notes_rule_code"]:
        if k not in df.columns: raise SystemExit(f"[ERROR] 缺少必要列: {k}")
    out["paper_id"]=df["paper_id"].apply(coerce_int)
    out["MethodTag_codes"]=apply_map(df["MethodTag"], MAP_MethodTag, "MethodTag")
    out["MethodFamily_codes"]=apply_map(df["MethodFamily"], mf_map, "MethodFamily")
    out["Methodology_codes"]=apply_map(df["Methodology"], mo_map, "Methodology")
    out["MethodTransparency_codes"]=apply_map(df["MethodTransparency"], MAP_MethodTransparency, "MethodTransparency")
    out["MethodNormativity_codes"]=apply_map(df["MethodNormativity"], no_map, "MethodNormativity")
    def norm_rule(v):
        iv=coerce_int(v)
        if iv is None: return None
        if 901<=iv<=909: return iv
        return None
    out["coder_notes_rule_code"]=df["coder_notes_rule_code"].apply(norm_rule)
    return out

def build_conflicts(long_df, out_path):
    recs=[]
    var_cols=VAR_ORDER[1:-1]
    for pid, sub in long_df.groupby("paper_id"):
        row={"paper_id":pid}
        for v in var_cols:
            codes=[x for x in sub[v].tolist() if x is not None]
            uniq=sorted(set(codes))
            row[f"{v}_agents"]=int(sub[v].notna().sum())
            row[f"{v}_unique"]=len(uniq)
            row[f"{v}_codes"]="|".join(map(str,uniq)) if uniq else ""
        rules=[x for x in sub["coder_notes_rule_code"].tolist() if x is not None]
        if rules:
            cnt=Counter(rules); mode=cnt.most_common(1)[0][0]
            row["rule_mode"]=mode
            row["rule_all"]="|".join(map(str,sorted(set(rules))))
        else:
            row["rule_mode"]=""; row["rule_all"]=""
        recs.append(row)
    out=pd.DataFrame.from_records(recs)
    out.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
    print(f"[OK] 写出冲突矩阵：{out_path}")

def krippendorff_alpha_nominal(long_df, out_path=None):
    def alpha_for_var(var_col):
        coders=sorted(long_df["agent"].unique())
        unit_ids=sorted([u for u in long_df["paper_id"].dropna().unique()], key=lambda x:int(x))
        mat=[]
        for pid in unit_ids:
            row=[]
            sub=long_df[long_df["paper_id"]==pid]
            for a in coders:
                val=sub.loc[sub["agent"]==a, var_col]
                v=val.iloc[0] if len(val)>0 else None
                row.append(v if pd.notna(v) else None)
            mat.append(row)
        import numpy as np
        from collections import Counter
        cats=set()
        for r in mat:
            for v in r:
                if v is not None: cats.add(v)
        cats=sorted(cats)
        if not cats: return float("nan")
        idx={c:i for i,c in enumerate(cats)}
        C=np.zeros((len(cats),len(cats)),dtype=float)
        for r in mat:
            vals=[v for v in r if v is not None]
            if len(vals)<2: continue
            cnt=Counter(vals)
            for i,ci in enumerate(cats):
                ni=cnt.get(ci,0)
                if ni==0: continue
                for j,cj in enumerate(cats):
                    nj=cnt.get(cj,0)
                    if i==j: C[i,i]+=ni*(ni-1)
                    else: C[i,j]+=ni*nj
        if C.sum()==0: return float("nan")
        Do=C.sum()-C.trace()
        marg=C.sum(axis=0)
        De=marg.sum()**2-(marg**2).sum()
        if De==0: return float("nan")
        return 1-Do/De

    vars_eval=["MethodTag_codes","MethodFamily_codes","Methodology_codes","MethodTransparency_codes","MethodNormativity_codes"]
    rows=[{"variable":v,"krippendorff_alpha_nominal": (lambda a: round(a,4) if a==a else "")(alpha_for_var(v))} for v in vars_eval]
    df=pd.DataFrame(rows)
    if out_path:
        df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
        print(f"[OK] 写出 α 报告：{out_path}")
    else:
        print(df.to_string(index=False))

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd")
    ap_s=sub.add_parser("sanitize", help="数值化并锁定列序")
    ap_s.add_argument("--in", dest="in_path", required=True)
    ap_s.add_argument("--out", dest="out_path", required=False)
    ap_c=sub.add_parser("conflicts", help="合并多Agent并生成冲突矩阵")
    ap_c.add_argument("--in", dest="in_paths", nargs="+", required=True)
    ap_c.add_argument("--out", dest="out_path", required=True)
    ap_a=sub.add_parser("alpha", help="快速α（名义）")
    ap_a.add_argument("--in", dest="in_paths", nargs="+", required=True)
    ap_a.add_argument("--out", dest="out_path", required=False)
    args=ap.parse_args()
    if args.cmd=="sanitize": sanitize_one(args.in_path, args.out_path)
    elif args.cmd=="conflicts":
        long_df=stack_agents(args.in_paths); build_conflicts(long_df, args.out_path)
    elif args.cmd=="alpha":
        long_df=stack_agents(args.in_paths); krippendorff_alpha_nominal(long_df, args.out_path)
    else: ap.print_help()

if __name__=="__main__": main()
