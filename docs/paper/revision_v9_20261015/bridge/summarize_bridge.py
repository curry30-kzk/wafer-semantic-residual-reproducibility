"""Summarize completed, audited bridge records without modifying any input."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
ROOT = PROJECT / 'results/teacher_review_v9_20261015/clap_bridge'
CODE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, payload):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf8')


def write_csv(path, rows):
    if path.exists():
        raise FileExistsError(path)
    with path.open('w', encoding='utf8', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def main():
    audit = read(ROOT / 'audit.json')
    if not audit['all_inputs_unchanged'] or audit['max_metric_recomputation_error'] != 0:
        raise ValueError('audited completion required')
    receipt = read(ROOT / 'freeze_receipt.json')
    budget = []
    for method in ('clap', 'ours'):
        rows = [read(p) for p in sorted((ROOT / 'validation' / method).rglob('*.json'))]
        if len(rows) != 320:
            raise ValueError('expected exactly 320 Validation records per method')
        budget.append({'stage': 'Validation', 'dataset': 'fgvc', 'method': method,
            'candidate_count': 16, 'support_tasks': 20, 'method_task_fits': len(rows),
            'optimizer_steps': sum(r['training']['optimizer_steps'] if method == 'clap' else r['training_steps'] for r in rows),
            'elapsed_seconds': sum(r['elapsed_seconds'] for r in rows),
            'learned_adaptation_parameters_per_task': 51200 if method == 'clap' else 100,
            'label_access': 'same 3333-image Validation set, 20 support tasks x 16 candidates',
            'prior_cached_visual_anchor_fit_cost_included': False,
            'prior_cached_feature_extraction_cost_included': False})
    for ds in ('wm', 'fgvc'):
        for method in ('clap', 'ours'):
            rows = [read(p) for p in sorted((ROOT / 'test' / ds / method).glob('*.json'))]
            if len(rows) != 20:
                raise ValueError('expected exactly 20 Test records per method/dataset')
            budget.append({'stage': 'Test', 'dataset': ds, 'method': method,
                'candidate_count': 1, 'support_tasks': 20, 'method_task_fits': len(rows),
                'optimizer_steps': sum(r['training']['optimizer_steps'] if method == 'clap' else r['training_steps'] for r in rows),
                'elapsed_seconds': sum(r['elapsed_seconds'] for r in rows),
                'learned_adaptation_parameters_per_task': (8 if ds == 'wm' else 100) * (512 if method == 'clap' else 1),
                'label_access': 'Test metrics evaluated after this freeze; earlier read-only Test label identity checks disclosed; no Test metrics used for candidate selection',
                'prior_cached_visual_anchor_fit_cost_included': False,
                'prior_cached_feature_extraction_cost_included': False})
    write_csv(ROOT / 'budget.csv', budget)
    write_json(ROOT / 'budget.json', budget)
    write_csv(ROOT / 'summary.csv', read(ROOT / 'summary.json'))
    extended = read(ROOT / 'summary_extended.json')
    report = [
        '# v9 CLAP 共享文本桥接实际结果', '',
        '本报告只描述完成后的追加事后分析，不替代原主表。两个 Test 都曾被历史开发观察；本轮冻结顺序不能恢复独立性。', '',
        'FGVC 双方共享冻结 T、同一套 20 个身份匹配的 Validation 支持任务与查询集，分别评估 16 个预先确定候选，以同一平均 Macro-F1 共同选择一个配置。匹配的是候选数和 Validation 任务访问，不是更新次数、参数量或总计算量。', '',
        f"- FGVC CLAP 选择：`{receipt['selected_fgvc']['clap']['config']}`；Validation 平均 Macro-F1 = {100*receipt['selected_fgvc']['clap']['mean_macro_f1']:.4f}%。",
        f"- FGVC 本文控制选择：`{receipt['selected_fgvc']['ours']['config']}`；Validation 平均 Macro-F1 = {100*receipt['selected_fgvc']['ours']['mean_macro_f1']:.4f}%。",
        '- WM CLAP 固定外部默认学习率 0.1／300 轮；本文固定既有 lambda 1／rho 0.01，不声称历史开发预算相同。', '',
        '| 数据 | shot | 本文F1均值±样本SD（%） | CLAP F1均值±样本SD（%） | F1差值／名义95%区间（pp） | Accuracy差值／名义95%区间（pp） | F1胜／平／负 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in extended:
        report.append(f"| {r['dataset']} | {r['shot']} | {r['ours_mean_percent']:.2f}±{r['ours_sd_percent']:.2f} | {r['clap_mean_percent']:.2f}±{r['clap_sd_percent']:.2f} | {r['difference_pp']:+.2f} [{r['ci95_low_pp']:+.2f}, {r['ci95_high_pp']:+.2f}] | {r['accuracy_difference_pp']:+.2f} [{r['accuracy_ci95_low_pp']:+.2f}, {r['accuracy_ci95_high_pp']:+.2f}] | {r['wins']}/{r['ties']}/{r['losses']} |")
    report += ['', '## 实际开发与更新预算', '',
        '| 阶段 | 数据 | 方法 | 候选 | 拟合任务 | 优化器更新数 | 记录总用时（s） |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in budget:
        report.append(f"| {r['stage']} | {r['dataset']} | {r['method']} | {r['candidate_count']} | {r['method_task_fits']} | {r['optimizer_steps']} | {r['elapsed_seconds']:.2f} |")
    report += ['', '上表是本轮记录的拟合／评价时间，不包含已缓存的图像特征提取，也不包含本文方法先前视觉锚点及其 OOF 的训练成本；不能据此直接断言完整系统速度优劣。CLAP 的 `train_seconds` 另存于逐任务记录，本文 `elapsed_seconds` 包含系数拟合、查询预测及指标计算，计时边界不同。', '',
        '## 身份、冻结与统计核验', '',
        '- 旧 FGVC Validation seed 5–9 的 10 个集群任务清单 SHA 不符；本轮统一使用 2026-09-22 依本机清单重建的全 20 个本地缓存。明细为 `validation_identity_resolution.json`，未混用旧任务，也未覆盖旧缓存。',
        '- 640 个 Validation 方法任务、80 个 Test 方法任务完整；全部保存指标从预测重算误差为 0。',
        '- 276 个受保护输入 SHA 不变；本轮冻结 receipt 早于本轮 Test 评价开始。',
        '- 冻结前曾只读 Test 标签核对任务身份并对文件计算 SHA；候选选择仅用 Validation 指标，不能声称冻结前从未访问 Test。',
        '- 保存的双方公共原始语义支持／查询先验数组逐 bit 相等。它们用于审计共同 T；CLAP 内部置信度计算仍保留原算法，不宣称全部内部浮点路径相同。',
        '- CLAP 使用 ProKeR 视觉特征缓存；本文保留原视觉分数缓存。60 对任务的清单／顺序、编码器及预处理标识已核验；不声称本文历史底层图像特征与 CLAP 逐 bit 相同。',
        '- 配对单位为 10 个支持采样种子，查询集固定；区间仅反映支持采样变化。配对 t 区间和精确符号翻转 p 均为名义统计，未做多重比较校正。',
        '- 原主表及历史文件保持原状。该桥接是新控制，不能称旧主表数值重现。', '',
        '完整目标、官方提交和移植差异见同目录 `PROTOCOL_NOTES.md`；执行脚本为 `run_clap_bridge.py`。']
    p = CODE / 'RESULTS_REPORT.md'
    if p.exists():
        raise FileExistsError(p)
    p.write_text('\n'.join(report) + '\n', encoding='utf8')
    output_files = [ROOT / x for x in ('audit.json', 'summary.json', 'summary.csv', 'summary_extended.json', 'summary_extended.csv', 'budget.json', 'budget.csv', 'protocol.json', 'freeze_receipt.json', 'validation_identity_resolution.json', 'input_metadata_audit.json', 'recompute_audit.json')]
    output_files += [CODE / x for x in ('PROTOCOL_NOTES.md', 'RESULTS_REPORT.md', 'ACCESS_AND_IMPLEMENTATION_AUDIT.md', 'test_bridge.py', 'test_recompute.py', 'recompute_bridge.py')]
    output_files += [Path(__file__)]
    write_json(ROOT / 'delivery_manifest.json', {
        'sha256': {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output_files},
        'all_results_reported': True, 'historically_unseen_test': False})
    print(p)


if __name__ == '__main__':
    main()
