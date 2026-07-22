# Ramen 瓶颈与非工程创新方向

**Query:** 仔细分析当前实验结果，判断还有什么非工程上的创新点可以尝试来提高 Ramen
**Generated:** 2026-07-21 16:36 CST
**Mode:** focused-iteration + literature + local-evidence synthesis
**Scope:** A20-A45 本地/服务器实验证据；2024-2026 一手论文；不使用 Ramen 测试 query 标签训练或调参

## 1. 结论先行

当前 Ramen 的核心问题不是码本容量、量化误差、Token 数量、query 融合强度或普通局部平滑，而是：

> **当一个 2D mask 在某视角同时包含 bowl + noodles，当前方法把这个“集合观测”当成了一个精确实体标签。污染在 Token 构建时已经发生，之后的路由、平滑和码本校准都只是在重排被污染的证据。**

最值得做的主创新是 **Set-valued Semantic Tomography with Occlusion Censoring（带遮挡删失的集合值语义层析）**：

1. 不再假设一个 SAM mask 对应一个 3D 实体；
2. 把每个重叠 SAM proposal 看成若干潜在 3D 实体在当前视角下的 union/partial observation；
3. 利用视角变化带来的“自然干预”分解粗粒度 mask：一些视角只暴露 noodles，另一些视角把 bowl + noodles 合并，它们的差异正是实体可辨识证据，不是应被丢弃的噪声；
4. 先恢复稳定实体归属，再对每个 Gaussian 挂载四个平等尺度 Token，最后重训四个码本。

这个方向的学术价值在于它改变了弱监督的统计假设，并可尝试给出“什么视角条件下实体可辨识”的命题；它不是又一个参数或后处理技巧。

## 2. 当前结果证明了什么

主数据来自 [完整对比文档](/Users/roxy/Desktop/drsplat_top_models_full_comparison.md) 和 [多轮实验 Summary](/Users/roxy/Desktop/report_methods_multiscene_summary.md)。

### 2.1 正式模型的 Ramen 差异是“尾部类别稳定性”，不是平均分不够高

| model | Ramen mIoU | Acc@0.25 | Acc@0.5 | 含义 |
|---|---:|---:|---:|---|
| aligned baseline A38 | 19.1030 | 23.9437 | 12.6761 | 单一静态语义严重不足 |
| A20 identity-preserving | 30.7216 | **45.0704** | **32.3944** | 实体/部件 identity 明显改善稳定性 |
| A33 four equal tokens | **30.8798** | 36.6197 | 29.5775 | mIoU 微升，但更多类别掉出严格阈值 |

A33 相对 A20 只增加 `+0.1582` mIoU，却损失 `-8.4507/-2.8169` Acc points。四 Token 让部分 query 找到了更好的尺度，但 hard-max 也让任意被污染的局部 Token 成为假阳性入口。

### 2.2 A20-A45 已经排除的解释

| evidence | observation | 排除的方向 |
|---|---|---|
| A24/A25 L2 micro | corn/nori 等局部目标受益，bowl 与 strict Acc 回退；point-supported attachment 几乎不变 | 不是 ID 挂载覆盖不足，而是独立 positive hypothesis 泄漏 |
| A28 feature MoE | 三场景均值只约 `28.4` mIoU | 不同尺度 CLIP feature 不能简单线性混合 |
| A36/A37 quantization confidence | 最佳只 `+0.0229` mIoU，Acc 不变 | 码本尺寸和量化误差不是主瓶颈 |
| A39 signed pair graph | `99.50%` 有效边为正边，Ramen `-0.093` mIoU | disjoint segment cache 在进图之前已丢失重叠、包含和互斥证据 |
| A40/A41 filtering | 全层过滤 `-1.290`；单 L0 `+0.524`，但 wavy noodles `-0.815` | 坏 mask 和有效细边界 mask 混在一起，硬删除不能分解它们 |
| A42-A44 scaffold | fixed-K 不跨场景泛化；HDBSCAN/core-residual 无标签 gate 失败 | 将已量化语义再聚类，不能恢复真实 entity density/granularity |
| A45 information gain | 路由改变 `11%-33%`，raw score 均值只改变约 `1e-3` 或更小；Waldo 安全门失败 | query-side 重归一化只在近等价码字间换位，不能恢复 object ownership |

### 2.3 类别级证据指向“集合标签错置”

- part/fine 表示可以显著提升 nori、napkin、glass of water，却让 wavy noodles 和 bowl 下降。
- L2 micro 能提升 corn 等小部件，但容器或相邻实体被 positive-max 污染。
- filtered L0 使 nori `+4.349`、plate `+4.159`、bowl `+0.725`，却使 wavy noodles `-0.815`。这是非常典型的 coarse union mask 与 fine component mask 冲突。
- A39 的正边平滑略帮助 chopsticks/napkin，却同时伤害 nori、sake cup、bowl、wavy noodles 和 glass of water。平滑不知道“同实体连续”与“两个接触实体”的区别。

这些结果的共同原因是：当 bowl + noodles 在某些视角被 SAM 合并，传统投票/平均/平滑把这个 mask 解读成“它们是同一个实体”。更正确的解读是“该 mask 仅说明其真实语义是若干潜在实体的集合，并未指明唯一实体”。

## 3. 为什么现有 SOTA 元素还不够

1. **Splat Feature Solver** 把 lifting 形式化为稀疏线性逆问题，并用 Post-Lifting Aggregation 丢弃与 3D cluster 不一致的 mask。它证明“在 lifting 前清洗观测”是关键，但 hard filtering 会丢掉包含有用细节的混合 mask；A40/A41 已在本项目中显示这一点。[Splat Feature Solver, ICLR 2026](https://arxiv.org/abs/2508.12216)
2. **CAGS** 明确研究跨视角粒度不一致，但它使用局部图传播和 mask-centroid InfoNCE。如果 mask 本身是 bowl + noodles，centroid 仍然是混合目标；低通平滑不会自动分解它。[CAGS](https://arxiv.org/abs/2504.11893)
3. **PairGS** 已经使用视角贡献和多视角 mask 证据构建 pair affinity，并用 TreeDBSCAN 形成层次聚类。因此“加 pair graph”本身不新；A39 也证明一旦 mask 被压成 disjoint segment ID，pair graph 会几乎只剩正边。[PairGS](https://arxiv.org/abs/2607.01140)
4. **LaGa** 已经对 object 保留多个视角 descriptor 并重加权，所以“每实体多 descriptor”更适合作为本方法的下游表示，不足以单独构成新主线。[LaGa, ICML 2025](https://proceedings.mlr.press/v267/cen25a.html)
5. **ReLaGS** 已经建立层次语义场景图并做关系推理；**ExtrinSplat** 已经建立重叠的 multi-granularity object groups 和文本假设。因此“场景图”或“重叠 group”也不能独立声称主创新。[ReLaGS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xie_ReLaGS_Relational_Language_Gaussian_Splatting_CVPR_2026_paper.html) [ExtrinSplat, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ding_ExtrinSplat_Decoupling_Geometry_and_Semantics_for_Open-Vocabulary_Understanding_in_3D_CVPR_2026_paper.html)
6. **MV3DIS** 用 3D guide 匹配跨视角 mask，并用深度一致性抑制遮挡歧义。它为“遮挡应该进入统计模型”提供支持，但其主体仍是 mask matching，而不是对 union mask 的潜在集合分解。[MV3DIS](https://arxiv.org/abs/2604.08916)

因此，真正可区分的问题定义是：

> **现有方法在“匹配、平滑、过滤或组织 mask”；新方法应该“把 mask 作为集合值弱监督，反演其潜在实体成分”。**

## 4. 主方向：Set-valued Semantic Tomography

### 4.1 潜在变量

设 `g` 是 Gaussian，`e` 是潜在原子实体，`v,p` 是视角和像素，`r` 是该视角的一个重叠 SAM proposal。

- `z_ge`: Gaussian `g` 属于原子实体 `e` 的概率；
- `w_vpg`: 已知的 3DGS alpha-compositing/view-contribution responsibility；
- `o_vpe = sum_g w_vpg z_ge`: 实体 `e` 在像素 `p` 的可见占用；
- `a_vre`: proposal `r` 在当前视角包含实体 `e` 的软 incidence；
- `d_e,h`: 实体 `e` 的第 `h` 个语义 descriptor，后续再投影到 L0-L3 四个平等 Token/Codebook。

### 4.2 集合观测似然

一个 mask 可以是多个实体的 union，用 noisy-OR 表示：

```text
P(M_vrp = 1)
  = 1 - product_e (1 - a_vre * o_vpe)
```

这与当前的差异是：当一个 mask 覆盖 bowl + noodles，模型不需要把两者压成同一个 embedding，而是让 `a_vr,bowl` 和 `a_vr,noodles` 同时为正。其他视角的 noodles-only mask 则约束 `a_vr,noodles`，从而将两者分解。

### 4.3 遮挡不是负标签，而是 censored observation

当某个 Gaussian 在另一物体后方时，当前像素 mask 不能说明它属于前景或背景。对每条 ray 引入潜在 visible owner，按 `T_i * alpha_i` 概率边缘化；未成为 visible owner 的后层 Gaussian 视为未观测，而不是继承前景 mask 语义。

这一步直接针对 Dr.Splat 式 top-K contributor lifting 的结构性污染：几何贡献可以用于似然，但不应把同一 pixel label 当成所有 ray contributor 的精确语义。

### 4.4 语义也应是 multiple-instance/set likelihood

对 mask 的 CLIP descriptor `y_vr`，不对其包含实体做 feature average，而是将它当成 positive bag：

```text
L_sem(v,r)
  = -log sum_e a_vre * exp(sim(y_vr, d_e) / tau)
```

可以为同一实体保留多个 `d_e,h`，但 descriptor 只能在实体归属分解后学习。这样可以吸收 LaGa 的多视角语义，又避免 A28 在 feature 层线性混合不同尺度。

### 4.5 从树改为 mereological hypergraph

Ramen 的 bowl/noodles 不只是“同/不同 instance”，还有 contains/inside/part-of/adjacent/occludes 关系。建议将 `a_vre` 形成的共现模式组织为 **mereological hypergraph**：

- 原子节点：bowl shell、noodles、nori、corn、chopsticks；
- 复合超边：bowl-with-contents、place-setting 等 union group；
- 有向关系：contains、inside、occludes；
- 互斥关系：同一可见表面上的 sibling ownership 竞争。

这不要求四 Token 存在父子优先级。L0-L3 仍然平等地与 query 比较；hypergraph 负责的是 **Token 产生前的实体归属** 和复合 query 的 union semantics，不是压制某一层分数。

## 5. 可写成理论贡献的部分

多视角因果表征研究已证明，多个部分可观测 view 可以提供单视角没有的潜变量可辨识性；多视角 object-centric work 也针对遮挡给出了 invariant object representation 的可辨识结果。[Multi-View Causal Representation Learning with Partial Observability, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/956e5427549a82a7472e02adc88360e9-Abstract-Conference.html) [Identifiable Object Representations under Spatial Ambiguities, ICML 2025](https://proceedings.mlr.press/v267/kori25a.html)

本项目可尝试证明一个更具体的命题（目前是研究目标，不是已证结论）：

> 在准确的渲染 responsibility、view sufficiency 和 proposal separability 下，如果每个原子实体都在至少一组视角/proposal 中提供区分于其他实体的 anchor support，且两个实体的 visibility-proposal incidence 不完全相同，则 noisy-OR 的 mask-entity 分解在 slot permutation 意义下可辨识。

这个命题可以从 separable non-negative factorization/Boolean matrix factorization 与部分可观测多视角表征两个方向建立。即使最终只能得到局部可辨识或充分条件，也比“新增一个 gate”更有方法论分量。

## 6. 其他可尝试的非工程创新

| priority | direction | 核心思想 | 与主方向关系 | 判断 |
|---:|---|---|---|---|
| 1 | **Entity-conditioned Conditional Information Bottleneck** | 最大化 `I(Token_l; scale-specific observation | entity)`，最小化 `I(Token_l; viewpoint | entity)` | 使四个平等 Token 分别保留粗粒度、物体、部件、边界的条件互补信息 | 值得作第二个方法贡献；要在 entity factorization 之后做 |
| 2 | **Risk-controlled Boundary Abstention** | 边界 Gaussian 输出 entity set/null，只在 odd/even 视角后验稳定时做硬归属 | 把“不确定”从量化误差改成 ownership posterior | 能降低 bowl/noodles 假阳性，但单独做可能只是校准模块 |
| 3 | **Amodal-Modal Semantic Cycle** | 对遮挡实体分开 amodal identity 和 modal visible fragment，要求两者跨视角 cycle consistency | SAMEO 等工作显示开放词汇 amodal mask 可恢复被遮挡整体 | 适合做补充 teacher，但生成式 amodal prior 可能幻觉，不应做唯一主线 |
| 4 | **Semantic Exclusion as Energy Conservation** | 在同一可见 surface 上，相邻 atomic entities 的 ownership mass 有限，而遮挡后层不参与前层竞争 | 为实体分解提供明确负证据，不再依赖 A39 稀疏负边 | 值得并入主方法的 loss，但单独不足以成为论文主创新 |

SAMEO 将 modal detector 与 amodal SAM decoder 结合，说明不可见轮廓可以作为显式潜变量，但本项目最安全的用法是将 amodal 预测当成一个可降权的先验，而不是真值。[Segment Anything, Even Occluded, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Tai_Segment_Anything_Even_Occluded_CVPR_2025_paper.pdf)

## 7. 建议的实验路线

### A46.0: 无标签可辨识性审计

先不训码本、不跑 mIoU，只用 Ramen 训练视角：

1. 恢复/保留 L0-L3 **重叠 SAM proposals**，不得只保留每像素唯一 `segment_id`。
2. 用 raw `T*alpha` 贡献构建 proposal-Gaussian soft incidence，并以 visible-owner 潜变量处理遮挡。
3. 在 odd/even views 独立拟合 noisy-OR/Boolean factorization，用 Hungarian matching 比较实体 slot 一致性。
4. 报告 held-out mask NLL/IoU、slot stability、mask-to-slot entropy、每 slot 双 split 支持视角数，以及“任意两 slot 是否存在至少一个分离 proposal”的 separability matrix。
5. 预先写死 gate：held-out mask NLL 相对单实体/hard-filter baseline 降低至少 `10%`；主 slot odd/even Jaccard 至少 `0.8`；每个候选 slot 在两个 split 均至少有 3 个支持视角。未通过就停止，不根据测试类别改 slot 数。

### A46.1: 连续语义机制验证

- 在通过 A46.0 的 entity slots 上学习 continuous multi-descriptor semantics，先不量化。
- 四尺度 descriptor 各自与 query 比较，只在 score 层选择，不恢复 A26 父子优先级。
- Ramen 机制门槛建议为：相对 A33 至少 `+2.0` mIoU，Acc@0.25 至少恢复到 A20 的 `45.07`，Acc@0.5 至少恢复到 `32.39`。只升 mIoU 但不恢复 strict Acc，说明仍然是少数大类改善掩盖尾部失败。

### A46.2: 四码本重训

只有 continuous 版通过后，才用相同 seed 重聚合 L0-L3 target，分别重训 `[2048,4096,8192,16384]` 码本。必须报告：

- continuous-to-codebook gap；
- 新码本 SHA 和 token assignment hash；
- 每个 entity 的 descriptor 数、slot occupancy 和存储；
- bowl/noodles、small-part、thin-object 三类错误分组。

### A46.3: 跨场景安全性

参数在 Ramen training-only diagnostics 后冻结，扩展 Figurines/Waldo/Teatime。正式 gate：

- 三场景平均 mIoU 至少 A33 `+0.5` points；
- 任一场景 mIoU 下降不超过 `0.5` points；
- Figurines/Waldo Acc@0.5 不下降；
- 至少 3 个 seed，并报告 entity-slot stability，不只报告最好 seed。

## 8. 关键消融：怎样证明不是另一个工程组合

1. `point label` vs `set-valued mask likelihood`：证明集合观测建模本身的作用。
2. `hard filter` vs `soft decomposition`：证明保留并分解冲突 mask 优于丢弃。
3. `all ray contributors inherit label` vs `occlusion-censored visible owner`：证明后层污染是一个独立机制。
4. `pair graph` vs `overlapping hyperedge incidence`：证明 A39 缺失的负/包含证据来自高阶关系，不是边权不够强。
5. `single descriptor` vs `entity-conditioned multi descriptor`：判断主收益来自 entity factorization 还是 LaGa 式多视角语义。
6. `continuous` vs `fresh codebook`：保持当前实验的量化合同，不把 target 改善与码本误差混在一起。

## 9. 不建议继续的方向

- 缩小 L3 码本、扩大某一层码本或再扫一组 K；
- 调 hard/soft fusion temperature、reliability weight 或 query threshold 作为主实验；
- 对已量化 token 继续 HDBSCAN/K-means 并期望恢复实体边界；
- 只用 disjoint `segment_ids` 再做一版 pair/signed graph；
- 用 query-side information gain、margin、LCB 或 codeword neighborhood 替代 token 归属学习；
- 仅加一个 VLM/MLLM 生成文本假设，却不改变 mask-to-entity 监督模型。

## 10. 总体判断

当前最值得的方向不是“在 A33 上再叠一个模块”，而是将 A33 的前端弱监督从 **mask-as-label** 重写为 **mask-as-set-valued projection**。四个平等 Token、四个独立码本和 query-score 层动态读取仍可保留，但它们必须建立在可辨识的实体归属上。

如果 A46.0 能在无测试标签条件下证明 latent slots 的双 split 稳定性，A46.1 又同时恢复 Ramen mIoU 和 A20 的 strict accuracy，那么这个方向有潜力成为“集合值多视角弱监督 + 遮挡可辨识性 + 层次语义内存”的方法论论文。若 A46.0 本身无法稳定分解实体，则说明现有 131 个训练视角/SAM proposals 不满足 separability；这时应停止场景内无监督微调，转向跨场景 object-centric prior 或 amodal teacher，而不是继续扫本地参数。

## 主要来源

- [Dr.Splat, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Jun-Seong_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.html)
- [Splat Feature Solver, ICLR 2026](https://arxiv.org/abs/2508.12216)
- [LaGa, ICML 2025](https://proceedings.mlr.press/v267/cen25a.html)
- [CAGS](https://arxiv.org/abs/2504.11893)
- [PairGS](https://arxiv.org/abs/2607.01140)
- [MV3DIS](https://arxiv.org/abs/2604.08916)
- [ReLaGS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xie_ReLaGS_Relational_Language_Gaussian_Splatting_CVPR_2026_paper.html)
- [ExtrinSplat, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ding_ExtrinSplat_Decoupling_Geometry_and_Semantics_for_Open-Vocabulary_Understanding_in_3D_CVPR_2026_paper.html)
- [Multi-View Causal Representation Learning with Partial Observability, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/956e5427549a82a7472e02adc88360e9-Abstract-Conference.html)
- [Identifiable Object Representations under Spatial Ambiguities, ICML 2025](https://proceedings.mlr.press/v267/kori25a.html)
- [Segment Anything, Even Occluded, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Tai_Segment_Anything_Even_Occluded_CVPR_2025_paper.pdf)

**Final confidence:** medium-high。对“当前瓶颈在 token 产生前的 entity ownership”信心高，因为 A24-A45 有多组独立反证；对“集合值层析一定能在 Ramen 上可辨识”信心中等，因为它依赖原始重叠 SAM proposals 和训练视角是否提供足够 anchor/separating evidence，这正是 A46.0 应先回答的问题。

## 11. A46 后续实验记录（2026-07-21）

本轮先完成了一个保守的 relation-level lower-bound audit，而不是第 7 节定义的完整 noisy-OR latent-slot factorization。输入是 preprocessing 已展平的四层 post-NMS `*_s.npy`，因此保留跨层重叠，但不保留同层原始 proposal overlaps。

### 11.1 主要结果

- 2,741,844 条多尺度有效有向 8-NN 边中，稳定 set-ambiguous 边只有 1,902 条（`0.06937%`）。
- 全局 held-out NLL：fixed relation `0.119709`，level-conditioned relation `0.167732`，后者相对恶化 `40.12%`。
- odd/even 关系符号一致率 `98.8194%`，说明失败不是随机不稳定，而是四层绝大部分局部边都给出相同的 interior relation。
- 用训练 split 选出的冲突边在另一 split 上评分，level-conditioned NLL 提升 `21.15%`，balanced NLL 提升 `22.59%`；尺度冲突是真信号，但覆盖极小。
- 将这些边作为四个平等 token 的 query-score 前置残差后，Ramen mIoU `30.8798 -> 30.8618`，Acc@0.25/0.5 不变；不通过 `+0.15` points micro gate。

### 11.2 更新后的判断

实验否定了“用 post-NMS 四层 partition + local pair graph 近似集合层析”的路线。8-NN 被物体内部正边淹没，即使边界冲突可预测，也只改变约千分之一 Gaussian，无法恢复 corn、kamaboko、onion segments 等完全缺失的实体语义。继续增大传播强度只会放大 sake cup 一类已有弱语义的误差，不会创造新的 ownership evidence。

这不构成对原始主假设的完整否定：raw overlapping proposals、proposal-Gaussian incidence、occlusion-censored visible owner 和 noisy-OR latent slots 均尚未实现。后续若重启该方向，必须直接执行第 7 节原始 A46.0，并将 held-out proposal mask likelihood/slot matching 作为 gate；不得再用局部 pair-sign 准确率替代高阶集合分解。当前按实验合同停止，不重训码本。
