# Verification records

The canonical saved-statistics replay is under `full_replay/`.
The original CLAP summary tables remain under
`../results/teacher_review_v9_20261015/clap_bridge/`; their extra verification
copies were identical. `CONSOLIDATED_OUTPUTS.json` maps all 16 removed copies to
retained files and records their content hashes.

These are saved verification receipts, not new training runs. Run
`python replay_all.py --output-dir recomputed/check` to produce a new replay.
The complete previous layout remains available in tag v1.2.0.
