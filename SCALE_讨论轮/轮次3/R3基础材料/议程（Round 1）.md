# R3 Round 1 议程（自动生成）

**范围**：优先级最高的 25 篇样本（见 `r3_priority.tsv` 前 25 行），逐项处理其存在分歧的变量。  
**排序逻辑**：按变量权重（Family 1.0 > Transparency 0.9 > Methodology 0.8 > Tag 0.7 > Normativity 0.6），再按 unique_count 决定先后。

## 总体指标
- 分歧篇数（80 篇）：
  - MethodFamily：73
  - MethodTransparency：64
  - Methodology：66
  - MethodTag：57
  - MethodNormativity：53
- Krippendorff’s α（名义）：
                variable  krippendorff_alpha_nominal
         MethodTag_codes                      0.2533
      MethodFamily_codes                      0.2833
       Methodology_codes                      0.2676
MethodTransparency_codes                      0.3161
 MethodNormativity_codes                      0.2917

## 操作步骤（每个议题）
1. 朗读：paper_id、变量名、四位 Agent 的当前编码（见 `r3_round1_ballots.tsv`）。
2. 证据：逐 Agent 提交最小证据句/段（仅 1–2 句）；缺失则用 `coder_notes_rule_code` 对应规则说明。
3. 讨论：先按“可证据升级/降级/合并”三类触发规则进行裁决；无法裁决者进入少数意见登记。
4. 落标：在 `r3_round1_ballots.tsv` 的 `decision` 列填入**统一编码**，并在 `proposed_rule_id` 标注所依据的规则。
5. 记录：把关键证据句转存到 `r3_discussion_log.md` 的对应议题下。

> 裁决模板：**优先 Family → Transparency → Methodology → Tag → Normativity**；若上位变量已定，尽量约束下位变量的合理空间。

