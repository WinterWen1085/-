# BN 节点与状态（v2）

- E_*（10 个）：二元 {0=无, 1=有}。
- MethodFamily：名义 {901,902,903,904,905,906,907,908,909,0(Unknown)}。
- method_class：字符串标签（保留历史口径），或在需要时转化为名义枚举。
- MethodTransparency：二元 {0,1}。
- PolicySalience / PoliticalSensitivity / SecrecyConstraint / DataAccess：有序离散 {0,1,2,3}。
- year：整数（若需，亦可分箱离散化）。
- topic：名义 {1,2,3,4}。

> 训练时建议两套输入并行对照：
> A) 仅要素层（10×二元）+ 情境 + 透明度；
> B) 要素 + 家族（或家族 Top-k 概率）+ 情境 + 透明度。