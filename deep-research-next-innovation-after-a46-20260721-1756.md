# A46 之后的架构创新方向：从语义 Token 到多假设实体层析

**Query:** 从整体架构以及实验结果上分析，判断目前还有什么创新性的方法能弥补当前缺点，可以从当前的 SOTA 论文以及其他类似领域寻找灵感
**Generated:** 2026-07-21 17:56 CST
**Mode:** focused-iteration + literature + local-evidence synthesis
**Scope:** 本地 A20-A46；2024-2026 开放词汇 3DGS、object-centric learning、video object segmentation、partial-label learning、amodal perception；不使用测试标签设计训练规则
**Confidence:** Medium-high

## 目录

1. [Executive Summary](#executive-summary)
2. [Key Findings](#key-findings)
3. [Evidence Ledger](#evidence-ledger)
4. [当前架构的结构性缺口](#当前架构的结构性缺口)
5. [SOTA 已经覆盖了什么](#sota-已经覆盖了什么)
6. [首选主创新：多假设实体语义层析](#首选主创新多假设实体语义层析)
7. [第二创新：语义归属驱动的 Gaussian Fission](#第二创新语义归属驱动的-gaussian-fission)
8. [理论贡献：视角可分性证书](#理论贡献视角可分性证书)
9. [备选方向与优先级](#备选方向与优先级)
10. [实验路线](#实验路线)
11. [Caveats And Contradictions](#caveats-and-contradictions)
12. [Further Research](#further-research)
13. [Sources](#sources)

## Executive Summary

当前模型的主要瓶颈不在四个 Token 是否平等、码本大小、量化误差或 query 融合公式，而在 **Token 产生之前的实体归属已经被错误确定**。A33 的三场景均值最高，但 Ramen 相对 A20 仅增加 `0.1582` mIoU，却损失 `8.4507/2.8169` Acc@0.25/0.5 points；A46 又证明跨尺度冲突在 held-out views 上确实可预测，但稳定冲突只占局部有效边 `0.06937%`，传播后 mIoU 反而下降 `0.0180` points。[1][2]

这组结果说明：

1. 四层 Token 提供了有用的语义多样性，但 hard max 让任何受污染 Token 都能成为假阳性入口。
2. `8-NN same/different` 只观察到实体边界，无法表达一个粗 mask 是 `bowl ∪ noodles` 的高阶集合事实。
3. post-NMS 平面 label map 已经丢掉同层 overlapping proposals；之后再做图、过滤、MoE 或码本校准都无法恢复丢失的 incidence 信息。
4. 当前固定 Gaussian geometry 也可能把两个接触实体压进同一批 primitive；纯语义重挂载无法修复这种 representational aliasing。

最值得推进的主创新是：

> **Occlusion-Censored Multi-Hypothesis Entity Tomography，遮挡删失的多假设实体语义层析。**

它不立即决定一个 mask 属于哪个实体，而是像 Multiple Hypothesis Tracking 一样保留若干 proposal-to-entity 关联路径；把 raw overlapping mask 当作潜在实体集合的 union observation；把遮挡后的 Gaussian 视为未观测而非负样本；只在后续视角提供足够 Bayes evidence 时才合并或排除假设。实体数量由 held-out reconstruction residual 自适应出生，而不是预设固定 slot 数。四个 L0-L3 Token 仍是平等语义读出，只是建立在更可靠的 entity posterior 上。

这个方向相对当前 SOTA 的可区分点是：Splat Feature Solver 解决线性 lifting 和 mask filtering，PairGS 解决 pair relation，ExtrinSplat 解决 overlapping groups/polysemy，LaGa 解决 view-dependent descriptors，Identity-aware LGS/Gaga 解决单路径 identity association；它们都没有把 **union mask、visibility censoring、delayed multi-hypothesis association 和 adaptive slot birth** 联合成一个可辨识的 3D 弱监督模型。[3][4][5][6][7][8]

## Key Findings

### 1. 当前瓶颈是 entity ownership，不是 semantic capacity

| 证据 | 实验观察 | 架构含义 |
| --- | --- | --- |
| A20 vs A33 | Ramen mIoU `30.7216 -> 30.8798`，但 Acc@0.25 `45.0704 -> 36.6197` | 多 Token 增加平均响应，但 identity-preserving 表示更能保护尾部类别 |
| A28 MoE | 多尺度 feature mixture 明显失败 | 污染 feature 的线性混合不会产生正确实体 |
| A36/A37 | 量化置信度最佳增益约 `0.023` mIoU | 码本误差不是主瓶颈 |
| A39/A46 | 局部关系图几乎全是正边；A46 stable conflict 仅 `0.06937%` | pair graph 被实体内部边淹没，无法表达 union/hyperedge |
| A40/A41 | 过滤部分 mask 有局部收益，但会伤害 noodles 等类别 | 混合 mask 同时含真信息和污染，硬删除不是分解 |
| A45 | 路由改变大量 Gaussian，但 raw score 约只变 `1e-3` | query 层只能重排近等价、已污染的码字 |

本地正式指标和实验合同见 [模型全对比文档](</Users/roxy/Desktop/drsplat_top_models_full_comparison.md>) 与 [实验 Summary](</Users/roxy/Desktop/report_methods_multiscene_summary.md>)。[1][2]

### 2. A46 失败没有否定 set-valued tomography

A46 测的是 post-NMS 四层 partition 上的 8-NN relation lower bound，不是 raw overlapping proposals 上的 noisy-OR latent factorization。它真正否定的是：

- 用局部 same/different 边近似集合观测；
- 用边界传播代替实体发现；
- 在已有 A33 Token 上做轻量 relation residual。

它没有检验：

- 同层 proposal overlaps；
- proposal-Gaussian incidence hypergraph；
- 可见 owner 与遮挡 Gaussian 的分离；
- variable-cardinality entity slots；
- delayed multi-hypothesis association。

### 3. “重叠 group”或“场景图”不能单独作为新主创新

ExtrinSplat 已将 Gaussian 聚成 multi-granularity overlapping object groups，并以外置文本假设支持 polysemy；ReLaGS 已建立 hierarchical language-distilled scene 和 semantic scene graph；PairGS 已构造 relation graph 和 hierarchical cluster tree。[4][5][9] 因此 mereological hypergraph 可以作为内部数据结构，但不能只凭“有重叠、有层次、有关系”宣称主创新。

## Evidence Ledger

| Claim | Evidence | Confidence | Caveat |
| --- | --- | --- | --- |
| A33 的 Ramen 尾部稳定性弱于 A20 | 本地严格对齐指标与 per-scene delta [1] | High | 仅三场景本地协议 |
| pair relation 对当前 Ramen 信号覆盖过小 | A46 held-out relation 与正式评测 [2] | High | A46 没保存 raw overlapping proposals |
| 当前 SOTA 已覆盖 inverse lifting/filtering | Splat Feature Solver [3] | High | 论文协议与本地协议不能直接比数值 |
| 当前 SOTA 已覆盖 overlapping groups/polysemy | ExtrinSplat [4] | High | 2026 新论文，复现成熟度尚待检查 |
| 当前 SOTA 已覆盖 pair graph/hierarchy | PairGS [5] | Medium-high | 2026-07-01 arXiv v1，尚需长期验证 |
| 单路径 identity alignment 已有强先例 | Identity-aware LGS、Gaga [6][8] | High | 仍依赖 tracker/pseudo-ID 质量 |
| 多假设 memory 可缓解一次错误的长期传播 | SAM2Long [13] | High | 原任务为视频 VOS，不是静态多视图 3D |
| 动态 slot birth 可由 reconstruction residual 驱动 | SlotCurri、Adaptive Slot Attention [14][15] | High | 迁移到 3DGS 需要新的 visibility likelihood |
| 多视图部分可观测变量存在可辨识条件 | ICLR 2024、ICML 2025 [11][12] | High | 理论不能直接套用，需给出 3D rendering 特化假设 |

## 当前架构的结构性缺口

### 1. Observation model 错把集合观测当精确标签

当前 preprocessing 在每个 SAM 层内部做 NMS，并把多个 mask 压成每像素唯一 `segment_id`。例如一个粗 proposal 同时覆盖 bowl 和 noodles 时，lifting 假设所有贡献 Gaussian 共享一个精确 group/feature。这个错误发生在 group 和 codebook 之前。

正确的统计对象应是：

```text
proposal r = one or more latent entities under one view
```

而不是：

```text
proposal r = exactly one entity label
```

### 2. Association model 过早坍缩

当前 dominant group、source gate 和后续 codebook assignment 都在较早阶段选择唯一解释。若某一视角把 bowl/noodles 合并，错误 identity 会进入后续所有聚合。Identity-aware LGS 使用 DEVA coherent identity labels、同/异 identity consistency 和 outlier filtering，Gaga 使用 3D-aware memory bank 做跨视图 mask association；它们证明 identity alignment 有效，但仍以单个 pseudo-ID 为主要路径。[6][8]

Ramen 需要的不是更强的单次匹配器，而是 **在证据不足时不提交唯一匹配**。

### 3. Visibility model 没有把“看不见”与“负证据”分开

raw top-K `T*alpha` 给出 ray contributor，但一个后层 Gaussian 即使贡献非零，也不应自动继承前景 mask 的语义；反过来，未落入前景 mask 也不代表它属于背景。遮挡区域本质上是 censored observation。

Splat Feature Solver 将 lifting 写成稀疏线性逆问题并通过 Tikhonov/aggregation 稳定求解，解决的是数值和 noisy feature；它没有将 occluded owner 作为离散潜变量。[3]

### 4. Entity cardinality 由前处理隐式决定

当前 Group 数量来自 SAM 分割和聚合规则。固定 slot 或固定 cluster 数容易在 coarse mask 下欠分，在纹理丰富区域过分。Adaptive Slot Attention 指出预先固定 slot 数是 object-centric model 的核心限制；SlotCurri 更进一步，仅在 reconstruction error 持续较高的位置分配新 slot，并用结构损失和前后循环抑制碎片化。[14][15]

### 5. 固定 geometry 可能形成语义不可分辨 primitive

如果同一 Gaussian 的投影支持跨过 bowl/noodles 或 spoon/background 边界，再好的 Token 也只能给这个 primitive 一个混合 ownership。Ilov3Splat 通过联合优化 geometry、language field 和 instance field说明 geometry-semantic joint optimization 是一条有效 SOTA 路径；MaskGaussian 则说明可以让暂未选中的概率 Gaussian 继续接收梯度，而不是一次性永久删除。[10][18]

当前模型缺少的是针对 **持续语义冲突** 的局部 Gaussian split/fission，而不是全场景重新 densify。

### 6. Query hard max 缺少 entity-level uncertainty

四个平等 Token 是合理设计，但 hard max 等价于“任何一层强响应即可通过”。当污染 Token 存在时，这会提高 recall，也会打开假阳性入口。A45 已证明在 codeword 层做 information gain/counterfactual normalization 不够，因为近邻码字语义几乎重复。风险控制只能放在 entity posterior 之后，不能替代 entity learning。

## SOTA 已经覆盖了什么

| SOTA direction | 已有贡献 | 本项目不能只做什么 | 尚未解决的空隙 |
| --- | --- | --- | --- |
| Splat Feature Solver [3] | sparse linear inverse lifting、理论误差界、mask filtering | 再做一个加权 closed-form lifting | conflicting mask 的集合分解与 visibility censoring |
| ExtrinSplat [4] | overlapping multi-granularity groups、polysemy、外置语义索引 | 只强调一个 Gaussian 属于多个 group | group 是怎样由 union masks 可辨识地恢复 |
| PairGS [5] | pair affinity、稀疏关系图、TreeDBSCAN hierarchy | 再做 same/different graph 或层次树 | proposal 级高阶 incidence、非局部 union relation |
| Identity-aware LGS [6] | tracker identity、same/different semantic consistency、boundary expansion | 加 identity contrastive loss | uncertain identity 应保留多路径而非立即 hard ID |
| LaGa [7] | object decomposition、multi-view descriptor clustering/reweighting | 每 object 多 descriptor | object decomposition 本身的弱监督归属错误 |
| Gaga [8] | 3D-aware memory bank 跨视图关联 | mask 与最大 overlap ID 做单路径关联 | 遮挡和 union mask 下的 delayed association |
| ReLaGS [9] | hierarchy、scene graph、relational retrieval | 添加 scene graph/GNN | graph 节点在构建前是否属于正确实体 |
| Ilov3Splat [10] | joint geometry/language/instance optimization | 普通 joint training | 由 ownership ambiguity 触发的局部 primitive fission |

## 首选主创新：多假设实体语义层析

建议方法名暂定为：

> **MHET-GS: Multi-Hypothesis Entity Tomography for Language Gaussian Splatting**

### 1. 核心思想

从 multi-object tracking 和 SAM2Long 借鉴 **Multiple Hypothesis Tracking**：不把每个新 mask 立即写入唯一 entity memory，而是维护若干低成本关联分支。SAM2Long 的 constrained tree memory 在不确定时保留多个可能 mask pathway，只有证据明确时才收敛，从而减少一次错误写入长期 memory 的影响。[13]

在静态多视图 3D 中，对应状态不是运动轨迹，而是：

- entity 的 Gaussian ownership；
- proposal-to-entity incidence；
- entity existence/visibility；
- entity 的四层 peer semantic descriptors。

### 2. 原始重叠观测

必须在 flatten 前保存每个视图、每个尺度的 raw proposal：

```text
R_v = {mask_vr, SAM confidence_vr, CLIP/DINO descriptor_vr}
```

每个 proposal 可以包含多个实体；每个 Gaussian 也可以属于 object、part、context 等多个尺度 group。这里的多个 group 不表示父子 Token 优先级，只表示观测 incidence。

### 3. 遮挡删失的可见 owner

设 `w_vpg` 为像素 `p` 对 Gaussian `g` 的 `T*alpha` responsibility，`z_ge` 为 Gaussian 对 atomic entity `e` 的 ownership，`c_vpg` 为可见 owner posterior：

```text
c_vpg ∝ w_vpg * P(g is first semantic surface on ray vp)
o_vpe = sum_g c_vpg * z_ge
```

未成为 visible owner 的 Gaussian 在该 ray 上记为 `unobserved`，不能继承前景语义，也不能成为背景负样本。

### 4. Set-valued mask likelihood

令 `a_vre` 表示 proposal `r` 在 view `v` 中包含 entity `e` 的概率。proposal mask 使用 noisy-OR：

```text
P(M_vrp = 1 | H)
  = 1 - product_e (1 - a_vre * o_vpe)
```

当 mask 是 bowl 与 noodles 的 union 时，`a_vr,bowl` 和 `a_vr,noodles` 可以同时为正；其他只观察 noodles 的 proposal 会在后续视角中提供 separating evidence。

### 5. 多假设关联树

每个 branch `H_b` 保存一组 proposal-to-slot assignments。遇到不确定 proposal 时执行三种 action：

1. `associate`：归入已有 entity；
2. `union`：由多个已有 entity 共同解释；
3. `birth`：创建新 entity slot。

branch 分数只用训练视图：

```text
J(H_b)
 = heldout mask NLL
 + lambda_sem * set-valued semantic NLL
 + lambda_cycle * cross-view cycle loss
 + lambda_mdl * model complexity
```

采用 beam/tree pruning 保留少量 posterior mass 最大且形状不同的路径；只有 Bayes factor、odd/even split 和 view support 同时满足阈值时才 collapse。这样避免单个污染 mask 永久写错 identity。

### 6. Partial-label purification

partial-label learning 把一个样本的监督定义为候选 label 集。NeurIPS 2024 的系统研究发现，成功方法的共同机制是将候选分布从 uniform 逐步净化到 one-hot，而不是一开始硬选。[16]

对应到这里：

- 初始 proposal 对多个 entity 保持软 incidence；
- 只有 anchor views/互斥共可见证据充分时降低 assignment entropy；
- 不可辨识的 composite proposal 永远可以保留为 union，不强迫 one-hot；
- 同时可见且空间互斥的 proposal 提供 verified negative，而非从遮挡像素制造假负样本。

### 7. 四个平等 Token 如何保留

entity posterior 稳定后，每个 entity 或 Gaussian 仍挂四个 L0-L3 peer Token：

```text
T_e = {t_e0, t_e1, t_e2, t_e3}
```

四个 Token 分别与 query 比较，不引入父 token 优先级。变化只在 spatial ownership：query score 先由 entity posterior 限定到可信支持域，再在四层 score 中动态读取。码本必须在连续 entity semantics 验证通过后重新训练，不能复用 A33 被旧 ownership 定义训练出的 codebook。

### 8. 为什么这可能改善 Ramen

- bowl/noodles：粗 mask 可由两个 entity union 解释，不再强迫相同 embedding。
- corn/kamaboko/onion：若存在至少一组独立可见 proposal，可通过 slot birth 从 bowl content 中分出。
- spoon/chopsticks：thin object 的 identity 可由跨视角 association 延续，而不只依赖局部 8-NN。
- sake cup/glass：遮挡后层不再继承前景 mask，减少稳定但错误的 positive evidence。

## 第二创新：语义归属驱动的 Gaussian Fission

### 1. 动机

多假设 association 仍假设现有 Gaussian 是足够细的空间 basis。若一个 Gaussian 在多个 view 中持续收到互斥 entity posterior，说明 semantic error 可能来自 primitive support 跨界，而不是 descriptor 不够好。

### 2. Fission criterion

仅对满足以下条件的 Gaussian 候选做 split：

```text
ownership entropy high
AND odd/even conflict stable
AND conflict appears in >= 3 views
AND projected residual has two spatial/depth modes
```

将父 Gaussian 分成少量 child primitives：

- geometry 初始化沿 covariance 主轴或 depth-responsibility modes；
- opacity 总质量守恒；
- child 分别接收不同 entity responsibility；
- 未被选中的 child 仍通过概率 masked rasterization 接收梯度。

MaskGaussian 的 probabilistic existence 和 masked rasterization 证明“暂未激活的 Gaussian 继续接受梯度”在 3DGS 中是可实现的；本方法将这一思想从 compression 转向 semantic ownership disambiguation。[18]

### 3. 与普通 densification 的区别

普通 densification 由 RGB reconstruction gradient 驱动；Semantic Fission 由 **跨视图、双 split 稳定的实体互斥 posterior** 驱动，目标是提高 semantic identifiability，而非 PSNR。该差异足以形成第二个方法贡献，但应作为 MHET-GS 的下游模块，而不是独立堆叠。

## 理论贡献：视角可分性证书

多视图因果表征研究证明，多个只观察部分 latent variables 的 view 可以识别单视图无法识别的共享与特有变量，并给出基于 view-subset 的 identifiability algebra。[11] ICML 2025 的 multi-view object-centric 方法进一步在 occlusion/view ambiguity 下，通过聚合 view-specific slots 学到 identifiable invariant object content。[12]

本项目可以将这些思想特化到 Gaussian rendering：

### 1. Proposed proposition

在以下条件下，entity ownership 在 slot permutation 意义下可辨识：

1. rendering responsibility 在可见 owner 上近似正确；
2. 每个 atomic entity 至少有一个 anchor proposal/view，不与所有其他 entity 永远共同出现；
3. 任意两个 entity 的 visibility-proposal incidence 列不完全相同；
4. union mask 由 noisy-OR 生成，且 model complexity 受限；
5. 遮挡 observation 被正确 censor，而不是当作负标签。

### 2. 可计算证书

训练后输出 `entity separability matrix`：

```text
Sep(e_i, e_j)
 = number / mass of views where one entity is supported without the other
```

并报告：

- anchor support；
- proposal-incidence matrix rank/condition；
- odd/even slot Hungarian match；
- heldout mask NLL；
- posterior entropy；
- unresolved entity pairs。

不能被观测分开的 pair 不应被模型自信地硬拆；应保留 set-valued output 或 abstain。这使模型能给出“哪些实体可从当前相机轨迹中恢复”的可审计结论，比继续调 gate 更接近论文级理论贡献。

## 备选方向与优先级

| Priority | Direction | Ramen fit | Novelty against SOTA | Risk | Verdict |
| ---: | --- | --- | --- | --- | --- |
| 1 | **MHET-GS: multi-hypothesis + censoring + set likelihood** | Very high | High | High implementation cost | 主线 |
| 2 | **Ownership-driven Gaussian Fission** | High for thin/contact objects | Medium-high | Geometry/token IDs must retrain | 主线第二模块 |
| 3 | **View separability certificate / curriculum** | High | High theoretical value | Proof assumptions may be strict | 主线理论部分 |
| 4 | Amodal-modal semantic cycle | Medium | Medium | Hallucination / query dependence | 仅作可降权 prior |
| 5 | Robust inverse lifting with censoring | Medium | Low-medium after SFS | Easier | 强 baseline/ablation，不作主创新 |
| 6 | Conformal or evidential query abstention | Low-medium | Medium | Cannot recover missing entities | 只作安全读出 |
| 7 | Cross-scene amortized entity prior | Potentially high | Medium-high | Requires more data and training | A47 失败后的后备方案 |

### Amodal prior 为什么只能做辅助

Open-World Amodal Appearance Completion 将 segmentation、occlusion analysis 和 inpainting 联合用于任意文本目标，说明 open-world amodal prior 已可用。[17] 但 Ramen 的目标是从已有训练视图恢复真实 3D ownership；生成式 amodal completion 可能把 plausible shape 当成真实证据。因此只允许它作为低权重 prior，并要求 modal-to-amodal-to-other-view cycle consistency，不能作为 pseudo-ground-truth。

### 为什么不建议 conformal/query calibration 做主线

它可以降低错误输出，但无法让 `corn/kamaboko/onion segments` 从 0 IoU 变成可检索实体。A36、A45 已经显示 query/quantization uncertainty 不是当前主要缺口。风险控制应在正确实体 posterior 之后使用。

## 实验路线

### A47.0: Raw Proposal Identifiability Audit

目标：只验证观测是否足以支持 entity factorization，不训练码本、不看 evaluation labels。

1. 重新运行 Ramen 131 个训练视图的 SAM，保存 pre-flatten overlapping proposals，使用 RLE/bitpack 和 proposal descriptor；不能只保存 `*_s.npy`。
2. 使用 raw top-45 `T*alpha` 构建 proposal-Gaussian soft incidence；显式计算 visible-owner posterior。
3. 比较四个模型：hard single-ID、PairGS-style pair baseline、single-path noisy-OR、multi-hypothesis noisy-OR + adaptive birth。
4. odd/even split 独立拟合，使用 Hungarian matching 对齐 slots。
5. 预注册 gate：
   - heldout mask NLL 相对 hard single-ID 至少降低 `10%`；
   - stable slots odd/even IoU/Jaccard 至少 `0.8`；
   - 每个 stable slot 每个 split 至少 3 个支持视图；
   - 非平凡 union/birth hypotheses 至少解释 `1%` heldout proposal mass，避免再次只命中 `0.069%` 边界；
   - unresolved slot pair 必须被证书标记，不能强行 hard assignment。

如果 A47.0 不过 gate，应停止场景内 unsupervised factorization，转向跨场景 object-centric prior；不要训练语义码本。

### A47.1: Continuous Entity Semantics

仅在 A47.0 通过后：

- 为 stable entity slots 聚合 Old/SAM semantic bags；
- 学习四个 peer continuous descriptors；
- query 与四层分别比较，entity posterior 只约束 spatial support；
- 暂不量化，隔离 representation gain 与 codebook error。

Ramen mechanism gate：

- 相对 A33 至少 `+2.0` mIoU points；
- Acc@0.25 至少恢复到 A20 的 `45.07%`；
- Acc@0.5 至少恢复到 A20 的 `32.39%`；
- corn/kamaboko/onion 中至少两个类别脱离 0 IoU；
- bowl/noodles 不允许只改善一个、显著伤害另一个。

### A47.2: Ownership-Driven Fission Ablation

只 split A47.1 中持续高 ownership entropy 的 Gaussian，比较：

1. no fission；
2. RGB densification control；
3. semantic fission；
4. semantic fission + probabilistic dormant children。

除语义指标外报告 RGB PSNR/SSIM、Gaussian count、split 区域比例和几何漂移，证明收益不是无约束增加 primitive 数量。

### A47.3: Fresh Four Codebooks

连续版本通过后，用 seed `20260719` 对四层 target 分别重新训练 `[2048, 4096, 8192, 16384]` codebooks。报告 continuous-to-quantized gap、codebook SHA、assignment hash、slot occupancy 和每 Gaussian bytes。

### A47.4: Cross-scene And Multi-seed

- 冻结参数后扩展 Figurines/Waldo/Teatime；
- 至少 3 个 seeds；
- 与 A33、A20、A38 及可复现 SOTA baseline 在同一 evaluation protocol 比较；
- 报告 entity-slot stability，而不是只报告最优 mIoU。

## Caveats And Contradictions

1. **外部论文 headline metric 不能和当前本地 mIoU 直接比较。** 不同工作使用不同 2D/3D selection、阈值、query 集和 calibration；本报告只比较方法假设，不据此声称当前模型已接近或远离某个 SOTA 数值。
2. PairGS 是 2026-07-01 的新 arXiv v1；其 SOTA 和 50x 声明来自作者论文，尚未经过长期复现。[5]
3. ExtrinSplat、ReLaGS 是 CVPR 2026 工作，已经显著压缩“overlapping groups/scene graph”作为独立创新的空间。[4][9]
4. 多视图 identifiability 理论通常假设 view mixing、latent sharing 或 slot generation 满足特定条件；Gaussian alpha compositing 和 SAM proposal noise 需要重新证明，不能直接引用结论替代推导。[11][12]
5. MHT tree 可能指数增长，必须使用 beam pruning、branch merge 和 MDL penalty；否则只是不可扩展的组合搜索。
6. Semantic Fission 会改变 Gaussian 数量和顺序，因此 A33 point token IDs/码本不能直接复用。需要从新 geometry contract 重建缓存、group 和 codebook。
7. raw overlapping SAM proposals 的存储和预处理成本明显高于当前 flat maps，但这是验证主假设所必需的信息，不应再为了方便提前丢弃。

## Further Research

在实现前还需完成三项针对性验证：

1. 阅读 ExtrinSplat/PairGS/Splat Feature Solver 的完整 supplementary 和代码，确认它们是否已包含 hidden soft multi-assignment、visibility censoring 或 branch hypotheses；当前公开摘要与方法说明未显示这些模块。
2. 对 Ramen 随机抽取 10-20 个训练视图，重新保存 raw SAM proposals，先测同层 overlap degree、union frequency 和 anchor proposal coverage；这一步可在完整 A47 前排除数据本身没有 separability 的风险。
3. 设计一个不使用类别 label 的 synthetic bowl-content benchmark，在已知 entity ownership 下检验 noisy-OR、MHT pruning 和 Gaussian fission 能否恢复 ground truth，作为理论与实现的单元实验。

## Sources

[1] [Dr.Splat 当前最佳模型、第二名与 Baseline 全指标比较](</Users/roxy/Desktop/drsplat_top_models_full_comparison.md>)，本地实验文档，更新于 2026-07-21。
[2] [A46 多尺度集合关系与稀疏边界残差](</Users/roxy/Desktop/report_methods_multiscene_summary.md>)，本地实验文档，2026-07-21。
[3] [Splat Feature Solver, ICLR 2026](https://openreview.net/forum?id=AepuXqQM4X)，稀疏线性 inverse lifting、Tikhonov Guidance、Post-Lifting Aggregation。
[4] [ExtrinSplat: Decoupling Geometry and Semantics, CVPR 2026](https://arxiv.org/abs/2509.22225)，multi-granularity overlapping groups 与 extrinsic textual hypotheses。
[5] [PairGS: Relation-Centric Open-Vocabulary 3D Gaussian Segmentation, 2026](https://arxiv.org/abs/2607.01140)，pair affinity、sparse graph、TreeDBSCAN hierarchy。
[6] [Identity-aware Language Gaussian Splatting, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Jang_Identity-aware_Language_Gaussian_Splatting_for_Open-vocabulary_3D_Semantic_Segmentation_ICCV_2025_paper.html)，identity-conditioned language consistency 与 progressive mask expansion。
[7] [LaGa: Tackling View-Dependent Semantics in 3D Language Gaussian Splatting, ICML 2025](https://proceedings.mlr.press/v267/cen25a.html)，object decomposition 与 view-aggregated multi-descriptor semantics。
[8] [Gaga: Group Any Gaussians via 3D-aware Memory Bank, TMLR 2026](https://www.gaga.gallery/)，基于 projected Gaussian overlap 的跨视图 mask association。
[9] [ReLaGS: Relational Language Gaussian Splatting, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xie_ReLaGS_Relational_Language_Gaussian_Splatting_CVPR_2026_paper.html)，hierarchical language scene 与 semantic scene graph。
[10] [Ilov3Splat: Instance-Level Open-Vocabulary 3D Scene Understanding, 2026](https://arxiv.org/abs/2605.04506)，joint geometry/language/instance field optimization。
[11] [Multi-View Causal Representation Learning with Partial Observability, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/956e5427549a82a7472e02adc88360e9-Abstract-Conference.html)，partial-view identifiability 与 identifiability algebra。
[12] [Identifiable Object Representations under Spatial Ambiguities, ICML 2025](https://icml.cc/virtual/2025/poster/44625)，occlusion/view ambiguity 下的 multi-view probabilistic slots。
[13] [SAM2Long: Enhancing SAM 2 with a Training-Free Memory Tree, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html)，MHT-inspired constrained tree memory。
[14] [Reconstruction-Guided Slot Curriculum, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Moon_Reconstruction-Guided_Slot_Curriculum_Addressing_Object_Over-Fragmentation_in_Video_Object-Centric_Learning_CVPR_2026_paper.html)，residual-driven slot allocation、structure loss、cyclic inference。
[15] [Adaptive Slot Attention, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_Adaptive_Slot_Attention_Object_Discovery_with_Dynamic_Slot_Number_CVPR_2024_paper.html)，dynamic slot number object discovery。
[16] [What Makes Partial-Label Learning Algorithms Effective?, NeurIPS 2024](https://openreview.net/forum?id=1LVTG7v689)，candidate-label purification 的经验设计原则。
[17] [Open-World Amodal Appearance Completion, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ao_Open-World_Amodal_Appearance_Completion_CVPR_2025_paper.html)，open-world segmentation、occlusion analysis 与 amodal completion。
[18] [MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_MaskGaussian_Adaptive_3D_Gaussian_Representation_from_Probabilistic_Masks_CVPR_2025_paper.html)，probabilistic Gaussian existence 与 masked rasterization。

**Final confidence:** medium-high。对“下一步必须改变 entity ownership observation/association model”信心高；对 raw Ramen views 是否满足足够 anchor separability 信心中等，必须由 A47.0 raw-proposal audit 决定；对 Semantic Fission 是否带来额外收益信心中等偏低，应在连续 entity semantics 通过后再做。
