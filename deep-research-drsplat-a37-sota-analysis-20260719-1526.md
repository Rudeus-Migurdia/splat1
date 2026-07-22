# Dr.Splat A37 当前结果、SOTA 对比与创新性审计

**Query:** 仔细分析当前实验结果，和当前SOTA论文比较，还存在什么缺点，有什么创新点
**Generated:** 2026-07-19 15:26 CST
**Mode:** literature + local-code audit + evidence refinement
**Scope:** 截至 2026-07-19；以 LERF-OVS 上直接选择 3D Gaussians、再渲染评测的开放词汇任务为主。论文数字仅在协议相近时横向参考，不把 2D rendered-feature segmentation、referring segmentation 和 3D object selection 混为同一榜单。

## 一、结论先行

当前方向**值得继续，但还不能宣称 SOTA，也不宜把 A37 的 quantization-LCB 当作主要创新**。

1. 当前真正有效的核心是 A30/A36 的 **四个平等多尺度 resident token + query 后在 score 层 late fusion**。它跨三个随机种子都高于 A20，平均增益约 `+0.687` mIoU points。
2. A37 的 within-level quantization percentile LCB 相对重新训练后的 hard max 只增加 `+0.0229` mIoU points，Acc 完全不变，属于很弱的附加证据。
3. A37 三场景正式结果为 `47.362 / 66.536 / 51.634`（mIoU / Acc@0.25 / Acc@0.5）。把 A31 teatime 的同架构结果拼入后，四场景 provisional mIoU 为 `52.539`，但它混用了不同 seed 和不同轮次，不能作为正式论文数字。
4. 相近 3D 协议下，当前公开方法的报告值已经到约 `60–65` mIoU。我们的 provisional `52.54` 高于 Dr.Splat、LightSplat 和 ProFuse 的论文值，但低于 OpenGaFF，并明显低于 PairGS、LaGa、ReLaGS、Splat Feature Solver。
5. 最有希望的论文定位不是泛称“层次语义”，而是：**Multi-Granular Resident Semantic Memory：每个 Gaussian 常驻多个独立离散尺度 token，查询时保持 token 对等，只在 score 层动态检索。** 单独的多尺度、codebook、hierarchy 或 query-aware retrieval 都已有先例，创新必须落在这一组合及其可验证收益上。

## 二、先澄清两个不同的“40/45”

这里有两个完全不同的参数轴：

- `Top-K=40/45`：每个像素射线用于语义注册的 Gaussian contributors 数量。Dr.Splat 论文报告 Top-40；本项目四个 baseline checkpoint 的 `cfg_args` 都明确为 `topk=45`，模型目录也叫 `*_topk45_weight_128`。因此本地 baseline 在 Top-K 上已经和当前实验对齐。
- `selection_threshold=0.40/0.45/0.50/0.55`：文本相关性得分筛选阈值。当前 A37 实际使用 `0.55`，历史 baseline 汇总使用 `0.50`；A38 才额外重测了 baseline 的多个 selection thresholds。

因此，A38 的 `0.40/0.45` 表不能解释为 Top-40/Top-45。若要复现论文 Top-40 baseline，需要另训或找到真正 `topk=40` 的 checkpoint；目前的本地 baseline 是 Top-45。

## 三、当前方法到底实现了什么

### 3.1 Resident memory

当前 builder 对 SAM L0-L3 构造嵌套 3D groups，并在每层用 Old 与对应 SAM level 的跨视角 split consistency 选择 group semantic source。四层分别训练独立 spherical K-means codebook，容量为 `2048/4096/8192/16384`。每个 Gaussian 固定保存四个 ID，而不是一个 512D feature。实现见 [build_seeded_hierarchical_resident_memory.py](/Users/roxy/Desktop/Dr-Splat/build_seeded_hierarchical_resident_memory.py:2)、[codebook 配置](/Users/roxy/Desktop/Dr-Splat/build_seeded_hierarchical_resident_memory.py:280) 和 [artifact layout](/Users/roxy/Desktop/Dr-Splat/build_seeded_hierarchical_resident_memory.py:591)。

形式上，每个 Gaussian `g` 保存：

```text
M(g) = {(z_g^0, r_g^0, u_g^0), ..., (z_g^3, r_g^3, u_g^3)}
```

其中 `z` 是层级 code ID，`r` 是 split/source reliability，`u` 是量化误差。

### 3.2 Query-aware score fusion

给定文本 query `q`，每层独立解码并计算：

```text
s_l(g, q) = similarity(C_l[z_g^l], q)
```

hard-max 让四层完全平等，取 adjusted score 最大的 token；没有父 token 优先级，也不先把四个语义 feature 平均。实现明确写着 “no hierarchy prior, per-level calibration, or preferred base representation”，见 [fuse_equal_query_tokens](/Users/roxy/Desktop/Dr-Splat/semantic_hypothesis_routing.py:407)。

A37 在此基础上把每层量化误差转成层内 percentile，并以 lower confidence bound 处理接近的候选；最终仍返回被选 token 的原始相关性得分，保持下游阈值尺度不变，见 [quantization-aware fusion](/Users/roxy/Desktop/Dr-Splat/semantic_hypothesis_routing.py:549) 和 [A37 runner](/Users/roxy/Desktop/Dr-Splat/scripts/run_a37_level_normalized_quantization.sh:38)。

## 四、本地实验结果审计

### 4.1 增益来自哪里

| 方法 | 三场景 mIoU | Acc@0.25 | Acc@0.5 | 判断 |
|---|---:|---:|---:|---|
| Dr.Splat local baseline，历史 `t=0.50` | 41.767 | 63.128 | 42.795 | Top-45，本地强 baseline |
| A20 fine identity | 46.294 | 66.052 | 49.867 | 四 token 前的强前驱 |
| A26 fixed hierarchical memory | 45.764 | 63.922 | 47.413 | 固定层级/门控下降 |
| A28 Old/L2/L3 feature-level MoE | 28.416 | 36.537 | 30.744 | feature mixing 严重失败 |
| A30 hard-max，seed 20260717 | 47.034 | 66.536 | 50.714 | score-level late fusion 有效 |
| A33 hard-max，seed 20260719 | 47.376 | 66.536 | 51.634 | 三个 seed 中最好 |
| A36 重训 hard-max | 47.339 | 66.536 | 51.634 | 重训后基本复现 |
| **A37 percentile-LCB** | **47.362** | **66.536** | **51.634** | 仅比 A36 `+0.0229` mIoU points |

这说明：

- A28 的 feature-level MoE 会把多尺度假设相互平均、污染或抵消；用户提出的“分别和 query 比较”方向是正确修正。
- A30 相对 A20 的主要增益来自保持四个语义假设独立，并在 query score 层才决策。
- A37 即使在约 `8.7%–16.4%` 的 covered points 上改变 token 选择，最终指标几乎不变，说明当前 codebook reconstruction error 与“语义选择是否正确”的相关性很弱，或正负变化互相抵消。

### 4.2 随机种子稳定性

三次独立 codebook seed 的 hard-max 结果：

| Seed | mIoU | Acc@0.25 | Acc@0.5 |
|---|---:|---:|---:|
| 20260717 | 47.034 | 66.536 | 50.714 |
| 20260718 | 46.535 | 66.066 | 48.604 |
| 20260719 | 47.376 | 66.536 | 51.634 |
| **mean ± population std** | **46.982 ± 0.345** | **66.379 ± 0.221** | **50.317 ± 1.269** |

因此论文主结果应报告 `46.98±0.35`，而不是只报告最佳 seed 的 `47.38`。尤其 Acc@0.5 的最差 seed 比 A20 低 `1.264` points，当前方法还不能声称严格准确率稳定提升。

### 4.3 hard max 不是所有场景都最优

- 三场景跨 seed 平均：hard max `46.982`，softmax `46.616`，hard max 更好 `+0.366`。
- teatime A31：softmax `68.938`，hard max `68.068`，softmax 反而高 `+0.870`。
- A35 的全码本 percentile/tail calibration 都下降；A36/A37 的量化置信也没有形成有效增益。

结论是四个 token 独立比较成立，但“永远 hard max”还不是稳定答案。当前缺的是语义正确性置信，而不是简单的 score 分布或量化误差校准。

### 4.4 场景瓶颈

| Scene | A37 mIoU | 主要判断 |
|---|---:|---|
| Figurines | 56.901 | 多尺度 token 对小物体/部件有效 |
| Ramen | **30.838** | 最大瓶颈；相邻实体、细长结构和边界假阳性仍严重 |
| Waldo Kitchen | 54.347 | 复杂场景有效，但仍有遮挡和碎片化 |
| Teatime | 68.068 | 来自 A31 hard-max，不是 A37 同 seed 正式结果 |

Ramen 不是码本容量问题。A24/A25 已显示 micro token 能帮助 corn、nori 等局部目标，但 positive-max 同时污染 bowl、wavy noodles 等邻接区域。缺失的是边界负证据、遮挡关系和邻接实体约束。

### 4.5 存储与覆盖率并没有表面上那么漂亮

仅四层 resident artifact：

| Scene | Resident artifact | 摊销 B/Gaussian | usable slot fraction | 至少一个 usable slot |
|---|---:|---:|---:|---:|
| Figurines | 50.39 MiB | 70.62 | 43.20% | 43.32% |
| Ramen | 49.80 MiB | 71.92 | 86.02% | 86.13% |
| Waldo | 85.28 MiB | 43.52 | 73.08% | 73.33% |

`covered_fraction=1.0` 只表示每点物理上写了四个 ID，不表示四个 ID 都可靠。Figurines 有约 `56.7%` 的 Gaussian 没有任何 usable resident slot，只能回退 A14 base。

A37 运行还依赖 A14 `base_ids + pruned_candidate_ids`。若把三个 artifact 的 manifest 存储简单相加且不做 codebook 去重，完整语义输入约为：

- Figurines `121.68 MiB`，约 `170.5 B/Gaussian`
- Ramen `124.31 MiB`，约 `179.5 B/Gaussian`
- Waldo `174.68 MiB`，约 `89.1 B/Gaussian`

所以当前“轻量化”只能说远小于逐 Gaussian 512D FP16 feature，不能说已经达到效率 SOTA。LightSplat 明确只注入 2-byte semantic index，并报告约 4.55 秒 feature distillation；我们的端到端缓存、四层 consensus、group 构建、K-means 和 A14 依赖尚未汇总。[LightSplat paper/project](https://vision3d-lab.github.io/lightsplat/)

## 五、和当前 SOTA 的数值比较

### 5.1 严格的协议警告

LERF-OVS 目前至少混有三种常被写成 “mIoU” 的任务：

1. 在 rendered language feature map 上做 2D segmentation；LangSplat 原论文整体 mIoU `51.4` 属于这一类。[LangSplat, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Qin_LangSplat_3D_Language_Gaussian_Splatting_CVPR_2024_paper.pdf)
2. 先在 3D 选择 Gaussians，再渲染 selected Gaussians 和 2D GT 比较；Dr.Splat、OpenGaussian、LightSplat、A37 属于这一类。
3. Ref-LERF/GeoCGA 的 referring segmentation，需要空间关系语言和图对齐，不能直接和普通类别 query 比分。[GeoCGA, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Tao_Geometry-Aware_Cross-Modal_Graph_Alignment_for_Referring_Segmentation_in_3D_Gaussian_CVPR_2026_paper.html)

Identity-aware LGS 报告的 `80.5` mIoU 是 novel-view 2D semantic mask，不是 Dr.Splat 式 3D object selection，因此不能说 A37 比它低 33 points。[Identity-aware LGS, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Jang_Identity-aware_Language_Gaussian_Splatting_for_Open-vocabulary_3D_Semantic_Segmentation_ICCV_2025_paper.pdf)

### 5.2 相近 3D protocol 下的参考表

| 方法 | 年份/状态 | LERF-OVS mean mIoU | 可比性与核心机制 |
|---|---|---:|---|
| Dr.Splat Top-40 | CVPR 2025 | 43.58 | 官方 3D object selection；直接 language registration + PQ。[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Jun-Seong_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.pdf) |
| Dr.Splat local Top-45 | 本项目 A38 | 46.11 | 四场景、baseline 最佳 `selection t=0.50`；内部直接基线 |
| LightSplat | arXiv/CVPR 2026 material | 47.58 | 2-byte index、mask filtering/clustering、4.55 秒。[paper](https://arxiv.org/abs/2603.24146) |
| ProFuse | CVPRW 2026 | 48.67 | context proposals + visibility-weighted uplifting + PQ。[paper](https://openaccess.thecvf.com/content/CVPR2026W/Viscale/papers/Chiou_ProFuse_Efficient_Open-Vocabulary_3D_Gaussian_Splatting_with_Early-Saturating_Semantic_Uplifting_CVPRW_2026_paper.pdf) |
| **A37 + A31 provisional** | 本项目 | **52.54** | `A37` 三场景加另一 seed 的 A31 teatime；不能作为正式四场景结果 |
| OpenGaFF | arXiv 2026 | 54.36 | geometry-conditioned feature field + structured codebook attention；与本方向重叠较强。[paper](https://arxiv.org/abs/2605.06088) |
| PairGS | arXiv 2026-07 | 60.4 | pairwise Gaussian relation graph + hierarchical cluster tree；其论文重实现基线与 canonical 表差异较大。[paper](https://arxiv.org/abs/2607.01140) |
| LaGa | ICML 2025 | 64.0 | object decomposition + view-aggregated semantic clustering/reweighting，解决 view-dependent semantics。[paper](https://proceedings.mlr.press/v267/cen25a.html) |
| ReLaGS | CVPR 2026 | 64.4 | training-free hierarchical language scene + 3D scene graph + relation reasoning。[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Xie_ReLaGS_Relational_Language_Gaussian_Splatting_CVPR_2026_paper.html) |
| Splat Feature Solver | ICLR 2026 | 65.1 | feature lifting 作为 sparse linear inverse problem；闭式解、理论误差界、post-aggregation filtering、自动阈值。[paper](https://openreview.net/pdf/dfff7bd48be6ddede4f6830214ceb1897f417b0e.pdf) |

这个表不能当作统一 leaderboard。Splat Feature Solver 在同一篇论文中同时给出 Dr.Splat 原论文 `43.6` 和其统一几何重实现 `49.5`，差 `5.9` points；LaGa 原报告与重实现也有差异。这恰好证明：没有统一 geometry、feature source、threshold 和 rasterizer 时，跨论文 1–3 points 的差距没有可信意义。

### 5.3 当前模型处于什么位置

- 相对 Dr.Splat paper `43.58`：provisional `+8.96` points。
- 相对本地 Top-45 baseline 最佳阈值 `46.11`：provisional `+6.43` points。
- 相对 LightSplat `47.58`：provisional `+4.96` points。
- 相对 ProFuse `48.67`：provisional `+3.87` points。
- 相对 OpenGaFF `54.36`：provisional `-1.82` points。
- 相对 LaGa/ReLaGS/SFS：约 `-11.5` 到 `-12.6` points。

在 A37 有正式结果的三个共同场景上，LaGa 平均约 `61.77`、Splat Feature Solver 约 `64.00`，A37 为 `47.36`。Ramen 上 A37 `30.84`，而 LaGa/SFS 分别约 `55.6/62.3`，是绝大部分差距来源。

## 六、哪些可以算创新，哪些不能单独算

### 6.1 可作为主创新的部分

**创新 1：每个 Gaussian 的多 resident discrete semantic memory。** 不是为每个 Gaussian 学一个 language embedding，也不是只挂一个 object cluster ID，而是固定挂四个尺度的离散语义候选。

**创新 2：每层独立 codebook，避免跨尺度量化竞争。** 粗粒度和细粒度 token 不在同一个词表里抢容量；当前使用 `2K/4K/8K/16K`，保证不同尺度有独立表达空间。

**创新 3：query-aware late fusion，而非 feature fusion。** 四个 token 分别与 query 比较，直到 score 层才选择或融合。A28 MoE 的大幅失败和 A30 的恢复，构成很清楚的机制证据：多尺度语义是多假设，不适合先平均成一个 feature。

**创新 4：训练期 split-consistency source gate。** 每个 group 在 Old 与对应 SAM source 中根据跨视角稳定性选择语义来源，而 query 阶段不再改变 source。这比单次投影更重视 multi-view reliability。

**创新 5：可审计、可重训的离散 artifact contract。** 固定 seed、独立 manifest、每层 K、ID dtype、量化误差和 vocabulary contract 都可验证。它属于重要工程贡献，但仅靠这一点不构成算法 SOTA。

### 6.2 不能单独声称新颖的部分

- **多尺度 language feature**：LangSplat 已有三个 semantic levels，并根据 query relevance 选择层级。
- **hierarchical grouping**：ReLaGS 和 PairGS 已明确构建层次场景/cluster tree。
- **codebook**：LEGaussians、Dr.Splat PQ、GALA、OpenGaFF 都在使用量化或 structured codebook。
- **query-aware codebook retrieval**：OpenGaFF 已使用 query 与 codebook entry 的 attention/similarity。
- **跨视角聚合**：LaGa、Splat Feature Solver、LightSplat 都直接处理 multi-view inconsistency。

因此最稳妥的 novelty claim 是组合创新：

> A per-Gaussian multi-resident, level-disentangled discrete semantic memory whose peer tokens remain separate until query-conditioned score-space retrieval.

不要写成“首个 hierarchical language Gaussian”或“首个 query-aware codebook”，这两种表述经不起 2025–2026 文献审查。

## 七、当前主要缺点

### 7.1 论文证据链不完整

1. A37 只有三个场景；teatime 来自另一 seed、另一轮 readout。
2. 当前报告偏向三次 seed 中最好的 `20260719`，而不是 mean±std。
3. A37 只测 `selection threshold=0.55`，没有训练外 calibration，也没有 threshold robustness。
4. baseline Top-45 与论文 Top-40 的关系虽已澄清，但还没有真正 Top-40 checkpoint 的同代码复现。
5. 没有在同 geometry、同 feature source、同 evaluator 下重跑一个 2025/2026 强 SOTA。

### 7.2 “层次”目前只存在于构建，不存在于推理逻辑

L1-L3 groups 在父 group 内构建，但 artifact 中 `group_parent_ids` 全为 `-1`，query reader 把四层当 peer candidates。这个设计符合“四个平等 token”的要求，却意味着当前模型实际上是 **multi-granular memory**，还不是 hierarchical reasoning：

- 没有 parent-child containment consistency；
- 没有 coarse-to-fine candidate restriction；
- 不理解 “mug handle” 这类 part-of relation；
- 不支持 “cup left of bottle” 等空间关系。

ReLaGS、PairGS、GeoCGA 已把 hierarchy/relations 做成显式图结构，这是当前明显差距。

### 7.3 hard-max 容易放大任意尺度的假阳性

只要一个细尺度 token 对 query 偶然高分，它就能覆盖其它三个尺度。A24/A25 和 Ramen 已证明这是实际问题。当前只有 positive candidates，没有：

- background/null expert；
- 邻接实体 signed negative evidence；
- occlusion/depth relation；
- query-conditioned boundary consistency。

### 7.4 量化误差不是语义不确定性

A37 的 chord reconstruction error 只测 codeword 离原 feature 多远，并不知道该 feature 是否来自错误 mask、错误 group 或错误物体。它不能替代 multi-view semantic uncertainty。A37 近零增益正是这个错位的实验证据。

### 7.5 per-scene codebook 与固定四槽效率不足

- 四套 K-means codebook 每场景重训，尚无 cross-scene reuse。
- 每点固定写四 ID，即使大部分 slot reliability 为零。
- Figurines 只有约 43% Gaussian 至少有一个 usable slot。
- 运行还依赖 A14 base/candidate artifacts，当前 storage 表若只报 A37 memory 会低估完整部署成本。
- 没有端到端 build time、peak VRAM、query latency 和 cache footprint 对照 LightSplat/SFS。

### 7.6 泛化和任务覆盖不足

目前没有 ScanNet、3D-OVS、Ref-LERF、大场景、稀疏视角和动态场景结果；也没有 synonym、属性、关系、part-of、negative/OOD query 测试。四个 LERF-OVS 场景不足以证明“开放词汇层次语义记忆”的普遍性。

## 八、下一步最有价值的路线

### 第一优先级：先把当前结果变成可信结果

1. 用 seed `20260719` 对 teatime 重建四层 codebook，跑 A36 raw max 与 A37，同一轮形成正式四场景表。
2. 三个 seed 全部跑四场景，报告 mean±std；预先固定 seed，禁止按结果选最好 seed。
3. 把 `Top-K` 与 `selection threshold` 写入每个 summary；若要对论文 Dr.Splat，新增真正 Top-40 checkpoint，不再靠目录名猜测。
4. 在不看 test GT 的 validation/training-view statistic 上确定 threshold，或采用 Splat Feature Solver 式 histogram auto-threshold；test sweep 只做敏感性分析。
5. 至少选择 OpenGaussian/LightSplat/LaGa 中一个，在相同 geometry、SAM/OpenCLIP、query 和 evaluator 下重跑。
6. 报告完整 runtime/storage：2D extraction、raw contributor cache、split consensus、group build、codebook train、peak VRAM、artifact disk、query latency。

### 第二优先级：让“hierarchical memory”真正成立

保持用户要求的四 token 平等比较，**不要恢复固定父优先级**。更好的做法是在 score 层之后增加 relation-aware correction：

```text
1. 四个 token 各自与 query 比较，得到四个原始 score；
2. 基于 multi-view co-membership、3D adjacency、depth ordering 建正/负边图；
3. 用图一致性修正 score，而不是融合 feature；
4. parent-child 只约束空间包含和边界，不决定哪个层先验更重要；
5. 最后动态选择 level 或 level combination。
```

这能保留当前最有效的 score-level late fusion，同时吸收 PairGS 的 pair relation、ReLaGS 的 scene graph 和 GeoCGA 的 geometry-aware alignment 优点。

### 第三优先级：把 semantic uncertainty 与 quantization uncertainty 分开

每个 slot 的置信应联合：

- train-view split agreement；
- mask/group stability；
- visibility 和 depth-order support；
- 邻接竞争 margin；
- codebook reconstruction error。

其中量化误差只做小修正，不能作为主置信。可使用 held-out training views 学一个 query-independent reliability estimator，再在 query score 上做 calibrated energy，而不是用 test 类别训练 gate。

### 第四优先级：更轻、更可泛化的 codebook

- cross-scene shared base codebook + scene residual codebook；
- 每 Gaussian 稀疏可变槽位，而不是固定四槽；
- 只为高 entropy/多尺度冲突点保留额外 token；
- 让 codebook capacity 由 rate-distortion 或 retrieval loss 决定，不固定按层翻倍。

## 九、最终判断

### 创新性评分

| 维度 | 判断 |
|---|---|
| 表示设计 | **较强**：四个 level-disentangled resident IDs/point 是清晰且可实现的设计 |
| 查询机制 | **中强**：score-level late fusion 有机制实验支撑，但与 LangSplat/OpenGaFF 有邻近思想 |
| 跨视角一致性 | **中等**：显式 split gate 有价值，但 LaGa/SFS 已有更系统方法 |
| 量化置信 | **弱**：当前没有有效涨点 |
| 效率贡献 | **尚未成立**：完整依赖和端到端成本未报告，明显弱于 LightSplat 的 2-byte/seconds 叙事 |
| SOTA 竞争力 | **中等**：大约处在 52–53 provisional 区间，距公开 64–65 仍有约 12 points |
| 论文完成度 | **偏低**：缺四场景同 seed、统一 SOTA 复现、统计和效率表 |

### 是否值得继续

值得，但应停止继续扫描 quantization-LCB、margin 或普通 softmax。下一轮的明确主线应是：

> **四个平等多尺度 token + relation-aware signed score correction + 真正四场景/多 seed/统一协议验证。**

如果 relation-aware score correction 能主要把 Ramen 从 `30.8` 提到 `40+`，同时不损失 Figurines/Waldo/Teatime，那么这个方向会从“有趣的多码本工程”变成有竞争力的方法论文。若 Ramen 仍无明显改善，则需要回到 feature lifting/3D grouping 本身，当前 query reader 已不是主要瓶颈。

## 主要来源

- [Dr.Splat, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Jun-Seong_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.pdf)
- [OpenGaussian, NeurIPS 2024](https://3d-aigc.github.io/OpenGaussian/)
- [LangSplat, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Qin_LangSplat_3D_Language_Gaussian_Splatting_CVPR_2024_paper.pdf)
- [LaGa, ICML 2025](https://proceedings.mlr.press/v267/cen25a.html)
- [LightSplat, 2026](https://arxiv.org/abs/2603.24146)
- [ProFuse, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/Viscale/html/Chiou_ProFuse_Efficient_Open-Vocabulary_3D_Gaussian_Splatting_with_Early-Saturating_Semantic_Uplifting_CVPRW_2026_paper.html)
- [Splat Feature Solver, ICLR 2026](https://openreview.net/pdf/dfff7bd48be6ddede4f6830214ceb1897f417b0e.pdf)
- [ReLaGS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xie_ReLaGS_Relational_Language_Gaussian_Splatting_CVPR_2026_paper.html)
- [GeoCGA, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Tao_Geometry-Aware_Cross-Modal_Graph_Alignment_for_Referring_Segmentation_in_3D_Gaussian_CVPR_2026_paper.html)
- [OpenGaFF, 2026](https://arxiv.org/abs/2605.06088)
- [PairGS, 2026](https://arxiv.org/abs/2607.01140)

**Confidence:** 本地实验与代码判断为高；跨论文绝对排名为中等，因为 geometry、feature source、threshold、2D/3D protocol 和官方/重实现结果存在显著差异。
