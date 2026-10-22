# Configuration and reproduction documentation for manuscript v19

This directory collects configuration tables, frozen prompt inputs and reproduction
instructions moved out of the English manuscript to improve readability. The v19
revision reorganizes existing material and adds no scientific evidence. Necessary
implementation constraints remain in a short Appendix A in the same manuscript;
Appendices B–D remain in that manuscript, without renumbering. No separate
supplementary PDF is introduced by this revision.

## Documents

- [Configuration and complete parameter tables](configuration.md)
- [All 32 WM prompts and 100 FGVC prompt-source assignments](semantic_inputs.md)
- [Reproduction levels, commands, coverage and limits](reproduction.md)
- [v18-to-v19 location map](manuscript_mapping.md)
- [Documentation source hashes](documentation_provenance.json)

## Release and evidence versions

The original public evidence snapshot was prepared for Chinese manuscript v14.
Later Chinese/English drafts v15–v18 and the v19 organization revision use these
same three saved-count evidence bundles for the results explicitly listed in
[the reproduction coverage table](reproduction.md#public-saved-count-bundles).
This does not claim that every table, figure or new analysis in every later draft
has its own executable replay here. Source folder version numbers are retained
as historical evidence identifiers.

Release v1.1.0 adds documentation only. Frozen configurations, prompts, code, task
records and numerical outputs from v1.0.0 remain byte-identical. The previous tag
v1.0.0 is retained. Main manuscript PDFs, unpublished correspondence and author
review notes are not distributed by this repository.

For a complete image-to-result rerun, additional provider datasets, model weights,
historical caches and preparation artifacts are required. The validated public
entry point is saved-confusion-count statistical recalculation, not one-command
end-to-end model reproduction.
