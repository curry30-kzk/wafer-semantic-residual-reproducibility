# v1.2.2 maintenance notes — 27 October 2026

This version is derived from v1.2.0 (commit
`21ca2f27aeaae28abca94e89bcc8fc0c0d9b9569`). All scientific input, configuration,
result and original freeze files are unchanged.

Changes:

- Correct the English Figure 2 legend from `Prescribed anchor` to `Visual anchor`.
  Portable input paths and installed-font fallback are retained.
- Add current-manuscript/material mapping, quick-start navigation and CITATION.cff.
  The software citation uses the existing public maintainer account.
- Remove 16 exact duplicate verification copies, retaining an identical authoritative
  file for each. `verification_outputs/CONSOLIDATED_OUTPUTS.json` records the mapping.
- Create a new public file manifest. Original historical SHA values are retained;
  changed maintenance files receive distinct new public identities.

The 310-file S2 cache asset is reused unchanged from v1.2.0. Previous tags and
release assets remain available. RIGHTS.md and upstream notices are unchanged.
No additional training, independent Test, software-author identity or DOI is claimed.
`VALIDATION_MAINTENANCE.json` reports fresh checks separately from prior validation.

Windows compatibility: v1.2.2 adds `.cff` to the explicit UTF-8 text-transport suffixes and a regression test. This accepts only exact declared LF/CRLF identities; numeric or binary changes remain rejected. The short-lived v1.2.1 release is retained unchanged.
