def q_prob(infer, var: str, evidence: dict, state_label: str) -> float:
    """
    返回 P(var == state_label | evidence) 作为 float。
    兼容离散标签顺序，避免 DiscreteFactor 下标错误。
    """
    # pgmpy 0.1.24+：show_progress 可传；旧版可去掉该参数
    q = infer.query(variables=[var], evidence=evidence, show_progress=False)

    # q 是 DiscreteFactor；从 state_names 里找目标状态的位置
    names = q.state_names[var]  # 例如 ['0','1','2','3'] 或 ['No','Yes'] 等
    try:
        idx = names.index(state_label)
    except ValueError:
        raise ValueError(f"状态 {state_label!r} 不在 {var!r} 的状态空间 {names!r} 中。"
                         "请确认离散化或标签映射一致（字符串/整数是否混用）。")
    return float(q.values[idx])
