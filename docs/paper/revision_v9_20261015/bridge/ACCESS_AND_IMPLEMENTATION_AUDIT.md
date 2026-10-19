# 本轮访问边界与共同 T 的实现审计

## 冻结前的访问边界

不能声称“本轮冻结前从未访问 Test 文件或标签”。本研究的两个 Test 集历史上已经被查看；除此之外，本轮 `test_bridge.py` 在冻结前读取已存 Test 标签与清单，只做任务身份、支持／查询顺序和折划分检查；`prepare` 还对 Test 相关文件计算 SHA。它们不是新的盲测。

准确陈述应为：**本轮 16 候选的配置选择仅使用 FGVC Validation 指标，没有使用 Test 指标；随后绑定选择 receipt，再执行本轮 Test 指标评价。**测试标签的只读身份检查和文件哈希访问不应被错误表述为零 Test 访问。本轮所有分析仍为事后比较。

## CLAP 如何实际使用同一 T

不能仅用“结果 NPZ 中附写了相同 semantic array”证明 CLAP 使用了共同语义。实际调用链如下：

1. `select`／`evaluate` 从 `text_fgvc.npz`／`text_wm.npz` 读取 `text` 数组，同一个对象也传给本文 `raw_scores`。
2. `fit(arr, text, config, ...)` 将该数组作为第四个实参传入未经更改的 `wm811k.clap.fit_clap_support_only(..., text_prototypes, ...)`。
3. 原 fitter 的 `_validated_inputs` 将它转为 float32、校验类数与维数并执行 L2 归一化。
4. `class_confidence_coefficients` 从该文本和同一支持增强特征计算真实类别零样本 softmax 置信度，并按类平均；类别约束系数没有被任意替换。
5. `text_tensor` 由该文本生成，`weights = Parameter(text_tensor.clone())` 以它初始化分类权重。
6. CLAP 训练损失包含 `(weights - text_tensor)^2` 的支持置信度加权类别 L2 惩罚。因此共同文本同时进入初始化、约束系数和约束中心。

桥接层只在调用原 fitter 的局部上下文中设置候选学习率，并传递轮数；结束或失败都恢复原模块学习率。默认学习率 0.1 的桥接调用已与直接调用原 fitter 做逐 bit 权重／类别系数及训练元数据一致性测试。

本文控制从相同 T 重新计算语义分数，再按自身原算法做支持校准、类别适配和启用判断。CLAP 保留其内部二次归一化、float64 softmax 等数值路径，不能声称所有内部语义分数和计算逐 bit 相同。

## 视觉输入的证据边界

CLAP 使用 ProKeR 已存的图像特征；本文语义分支也使用这些特征与共同 T，但本文视觉锚点保留其原 LP／OOF／query 分数缓存。两种方法采用不同分类头，不能强制 CLAP 接受本文视觉分数。

已核验的是当前支持清单与查询图像顺序、标签、类别顺序，以及视觉编码器／预处理标识。ProKeR 图像特征与共同 T 有相同模型状态 SHA；本文旧视觉锚点底层特征未全部保存，因此不能由相同标识推断其底层浮点数组逐 bit 相同，也不把旧 WM 独立锚点与内部锚点的批量差异抹掉。

## 数值和预算报告范围

保存的公共原始支持／查询 cosine 数组是共同先验的辅助审计产物；它们逐 bit 相同，但不是 CLAP 内部训练过程的替代证据。逐任务 CLAP JSON 另保存真实类别系数、轮数、学习率、batch count、optimizer steps、参数量与训练时间。

双方候选数和 Validation 访问任务一致，计算预算不同。本文参数数目仅指新增类别系数；完整系统还包含此前拟合的视觉锚点，不能以新增系数数量冒充完整系统训练参数或总计算成本。

`recompute_bridge.py` 只依赖 Python 标准库，从随包 80 个 Test JSON 的混淆矩阵重算指标和支持种子配对统计。它证明保存结果与汇总一致，不是从原图重新编码并完整训练的复现。
