# Optional historical cache archive

Extract over the matching source/evidence release root. The source package's
RELEASE_MANIFEST.json lists and verifies every required public cache byte.
Use a source version only when its RELEASE_MANIFEST.json declares these exact
310 cache files. Materials v1.2.1 intentionally reuses the unchanged v1.2.0 cache
asset; its content hashes are identical. The archive contains derived
features/scores and metadata, no original images or model checkpoint weights.
The verified rerun scope is classwise residual fitting for WM structured/Qwen
and FGVC main Test at5/10-shot. All-baseline retraining and image encoding are
outside that executed scope. See README.md for exact commands and limitations.
