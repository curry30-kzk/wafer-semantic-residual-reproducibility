# v9 共享文本与开发候选预算桥接

本实验是追加事后分析。WM 与 FGVC Test 均已被历史开发观察；本轮方案和选参冻结仅约束本轮新增比较，不能产生历史未见测试证据。原主表、旧配置和旧结果不被覆盖。

## 比较定义

- 只比较 5-shot、10-shot，各 seeds 0–9。
- 两个方法从同一冻结文本原型文件 `results/revision_v4_20261013/same_text_proker/text_{wm,fgvc}.npz` 获取 T，沿用相同提示组和原型构建规则。
- CLAP 复用 `results/proker_20261013/cache/` 中已有的冻结 CLIP ViT-B/32 clean、10-view support、query 特征；本文语义分支也由这些 clean／query 特征和共同 T 计算。不重新提取图像特征。
- CLAP 保留其独立的文本初始化分类权重、支持置信度类别约束和支持集交叉熵目标。CLAP 不使用本文方法的视觉锚点或语义校准机制。
- 本文新变体保留已归档的视觉支持／OOF／查询分数、折划分和 200-step Adam 目标，以同一 T 重新计算语义分数及支持校准。
- 本文视觉分类头与 CLAP 独立，底层历史图像特征未全部以数组保存，无法证明本文旧视觉锚点与 CLAP 当前输入在所有图像上逐 bit 相同。这里只核验支持清单、查询顺序、预处理和编码器标识，以及 ProKeR 图像特征与共同 T 的 checkpoint 状态 SHA 匹配；不声称双方共用同一视觉分数或底层图像浮点数组。
- CLAP 内部置信度计算保留原实现的归一化和 softmax 数值路径。因此“共享 T”不能进一步表述为两种算法所有内部浮点分数逐 bit 一致。

## FGVC：同一开发任务和候选数

CLAP 采用 16 个预设候选：学习率 `{0.01,0.03,0.1,0.3}` × 轮数 `{100,200,300,600}`，包含原移植配置 0.1／300。固定 SGD、momentum 0.9、weight decay 0、batch size 256、1 轮预热、预热学习率 1e-5、余弦调度、logit scale 100。每轮批次数仍为 `ceil(CK/256)`；不把 10-view 数据全部循环十遍。

本文新变体采用原有 16 个候选：lambda `{0.25,0.5,0.75,1}` × rho `{0.01,0.1,1,10}`，保留盒约束、支持 OOF 校准、收缩和任务启用规则。

双方分别在相同的 5／10-shot × 10 seeds 共 20 个 Validation 任务上，以平均 Macro-F1 共同选择一个配置；不能按 shot 独立选择，也不按 Test 表现选择。近似并列（1e-12）时取事先排列的第一个候选：CLAP 优先较少轮数、其次较小学习率；本文优先较大 rho、其次较小 lambda。完整候选结果均保留。

这里匹配的是**数据集级候选数量与 Validation 访问任务**。不同方法的每任务更新次数、参数量、OOF／视觉锚点成本、总训练时间均不相同，必须分别记录；不得称为相同训练计算预算。

16 候选预算只约束本轮新增适配网格；先前的方法设计、文本构建、视觉锚点选择和历史试验次数没有被重新随机化或重建。因此也不能把该追加控制升级为双方整个研究历程的等开发成本实验。

## Validation 旧缓存身份问题

预运行身份测试发现，`results/fgvc_aircraft_validation/class_reliability_residual/qwen2_5_7b_instruct/cache/` 中 5／10-shot、seed 5–9 共 10 个旧集群任务的 manifest SHA 与本机冻结清单不符。该情况已由 `docs/experiments/2026-09-22-semantic-source-matching.md` 披露，不能把旧视觉 OOF 分数与当前另一个支持任务的语义分数强行配对。

因此，本轮**全部 20 个 Validation 任务统一**使用已按本机清单重建的
`results/semantic_source_matching/fgvc_validation_runs/qwen2_5_7b_instruct/cache/`。
新选择使用这套身份匹配的开发任务；它不是旧主表 Validation 数值的重现。

每任务核验：当前清单文件 SHA = 本文缓存 manifest SHA；ProKeR 特征缓存的 manifest SHA = 同一清单 SHA；支持图像顺序、标签顺序、类别顺序一致；查询图像顺序和标签一致；视觉 OOF 与语义校准折一致；冻结文本与视觉编码器 checkpoint SHA 一致。相关 JSON／NPZ／清单和源码文件 SHA 已写入本轮 `protocol.json`（276 个文件），旧目录保持原状。

FGVC Test 与 WM 使用原主方法视觉缓存，身份测试全部通过；新控制不替换原主表。

## WM：只做共享文本比较

WM 没有可重新用于本轮公平选择的独立 Validation。CLAP 固定外部默认学习率 0.1、300 轮；本文固定既有 lambda 1、rho 0.01。不给任何一方按 WM Test 新选参。两者仅匹配文本、编码器及任务身份；不能宣称历史开发预算被重建或相等。

## CLAP 来源与变化

参考作者仓库提交 `dbdf4095b3d824546474c725539b9f4efff3ca0e`：

- [官方训练器](https://github.com/jusiro/CLAP/blob/dbdf4095b3d824546474c725539b9f4efff3ca0e/trainers/adapters.py)
- [官方主实验命令](https://github.com/jusiro/CLAP/blob/dbdf4095b3d824546474c725539b9f4efff3ca0e/experiments/experiment_CLAP.txt)
- [官方优化配置](https://github.com/jusiro/CLAP/blob/dbdf4095b3d824546474c725539b9f4efff3ca0e/configs/trainers/SGD_lr1e-1_B256_ep300.yaml)

本机 `src/wm811k/clap.py` 已按该公开主配置实现固定支持置信度的类别 L2 约束；未加入反复外层乘子更新。本轮通过局部恢复上下文只设置候选学习率，并传递候选轮数，调用原 fitter，不改变目标函数。

保留已有同协议移植差异：ViT-B/32（官方主命令 RN50）、10 个缓存视图（官方 20）、固定 logit scale 100、确定性的 PyTorch 支持重排。本轮新增 Validation 学习率／轮数搜索。因此应称“共享文本的 CLAP 同协议移植对照”，不能冒充官方数值或逐行复现。

## 冻结与验收

执行顺序为 `prepare → select → freeze → evaluate → audit`。`freeze_receipt.json` 绑定双方完整 Validation 网格、选择、协议和输入；只有 receipt 存在且所有文件 SHA 不变时才能开始 Test。全部 Test 结果无论有利或不利均保留。

核心测试已实际通过 10 项，包括默认 CLAP 调用等价、参数恢复、无查询标签拟合接口、任务身份、输入篡改阻断及非单位 lambda 的分数／预测一致性。最终报告还必须根据实际运行核对 640 个 Validation 方法任务、80 个 Test 方法任务、逐种子指标重算、共享 T 及保存的公共原始先验、全部历史输入 SHA，以及冻结与本轮 Test 的时间顺序。
