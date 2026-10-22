# Frozen semantic inputs

This is a readable index of adaptation inputs moved from v18 Appendix E. These
descriptions are not generated for individual query images and are not certified
by an independent expert rating study. Original source files and metadata are
unchanged in v1.1.0.

## WM: all 32 structured descriptions

Source: [`src/wm811k/llm_descriptions.py`](../../src/wm811k/llm_descriptions.py).
Version: `wm811k_codex_gpt5_structured_v1`. The original artifact records its source as `openai_codex_gpt5_family_design_session`.
The descriptions were produced once in a tool-assisted design session and then
frozen; that provenance string is retained verbatim in the source snapshot.

Each class has four descriptions, covering location, morphology, coverage and
confusable-pattern boundaries. Text prototypes use the paper's normalization,
averaging and renormalization procedure. Exact inputs follow verbatim.

### Center

- **Location:** A semiconductor wafer defect map with defective dies concentrated around the geometric center of the circular wafer.
- **Morphology:** A semiconductor wafer defect map containing one compact, roughly circular or irregular defect cluster in the middle.
- **Coverage:** A semiconductor wafer defect map whose central region is defective while most surrounding and edge regions remain clear.
- **Boundary:** A semiconductor wafer defect map with a centered cluster, not an annular donut, an outer-edge ring, or an off-center local cluster.

### Donut

- **Location:** A semiconductor wafer defect map with defects located in a band surrounding the center but away from the outer circumference.
- **Morphology:** A semiconductor wafer defect map whose defective dies form an annular, hollow, donut-shaped structure.
- **Coverage:** A semiconductor wafer defect map with a relatively clear center and outer area separated by a defective middle-radius band.
- **Boundary:** A semiconductor wafer defect map with an interior ring around the center, distinct from a solid central cluster or a ring on the wafer edge.

### Edge-Loc

- **Location:** A semiconductor wafer defect map with defects concentrated at one limited angular section of the outer wafer boundary.
- **Morphology:** A semiconductor wafer defect map containing a compact arc-shaped or blob-shaped local cluster that touches the edge.
- **Coverage:** A semiconductor wafer defect map where only a small portion of the circumference is defective and the remaining edge is mostly clear.
- **Boundary:** A semiconductor wafer defect map with a localized edge cluster rather than a continuous edge ring or an interior local cluster.

### Edge-Ring

- **Location:** A semiconductor wafer defect map with defective dies distributed along most of the outer circumference of the wafer.
- **Morphology:** A semiconductor wafer defect map whose defects form a continuous or nearly continuous circular ring at the boundary.
- **Coverage:** A semiconductor wafer defect map with high defect coverage around the perimeter while much of the inner wafer remains clearer.
- **Boundary:** A semiconductor wafer defect map with a circumference-wide edge band, distinct from one localized edge patch or an interior donut ring.

### Loc

- **Location:** A semiconductor wafer defect map with a defect concentration in one off-center interior region that does not touch the boundary.
- **Morphology:** A semiconductor wafer defect map containing one compact irregular blob or cluster inside the wafer.
- **Coverage:** A semiconductor wafer defect map where a limited interior area is defective and most other wafer sites remain clear.
- **Boundary:** A semiconductor wafer defect map with an interior localized cluster, distinct from a centered cluster, an edge-localized cluster, or random scatter.

### Random

- **Location:** A semiconductor wafer defect map with defective dies appearing at many unrelated locations across the wafer.
- **Morphology:** A semiconductor wafer defect map containing isolated points and small irregular fragments without a stable geometric shape.
- **Coverage:** A semiconductor wafer defect map with sparse or moderate defects dispersed broadly rather than concentrated in one region.
- **Boundary:** A semiconductor wafer defect map showing unstructured scatter, not a compact cluster, line-like scratch, ring, or near-full coverage.

### Scratch

- **Location:** A semiconductor wafer defect map with defective dies aligned across an elongated path through part of the wafer.
- **Morphology:** A semiconductor wafer defect map whose defects form a narrow linear, diagonal, curved, or streak-like structure.
- **Coverage:** A semiconductor wafer defect map with concentrated lengthwise coverage and relatively small width compared with the wafer diameter.
- **Boundary:** A semiconductor wafer defect map with a coherent line-like streak, distinct from random scattered points or a compact local blob.

### Near-full

- **Location:** A semiconductor wafer defect map with defective dies occupying almost all interior locations across the wafer.
- **Morphology:** A semiconductor wafer defect map forming a dense wafer-shaped mass rather than a localized geometric pattern.
- **Coverage:** A semiconductor wafer defect map with very high defect coverage and only a small number of clear sites remaining.
- **Boundary:** A semiconductor wafer defect map dominated by near-complete coverage, distinct from sparse random defects or any localized cluster or ring.

## FGVC: mixed-package assignments

The frozen JSON contains 100 class entries: 42 accepted constrained-Qwen outputs
and 58 deterministic hierarchy fallbacks. This is a mixed package, not 100 accepted
language-model descriptions. Generation/validation metadata, original responses
and fallback reasons remain in the JSON.

Source: [`data/processed/FGVC-Aircraft/semantics/qwen2_5_7b_instruct_v1.json`](../../data/processed/FGVC-Aircraft/semantics/qwen2_5_7b_instruct_v1.json).

| Class index | Variant | Final prompt source |
| --- | --- | --- |
| 0 | 707-320 | Accepted constrained-Qwen output |
| 1 | 727-200 | Accepted constrained-Qwen output |
| 2 | 737-200 | Accepted constrained-Qwen output |
| 3 | 737-300 | Accepted constrained-Qwen output |
| 4 | 737-400 | Accepted constrained-Qwen output |
| 5 | 737-500 | Accepted constrained-Qwen output |
| 6 | 737-600 | Accepted constrained-Qwen output |
| 7 | 737-700 | Accepted constrained-Qwen output |
| 8 | 737-800 | Accepted constrained-Qwen output |
| 9 | 737-900 | Accepted constrained-Qwen output |
| 10 | 747-100 | Deterministic hierarchy fallback |
| 11 | 747-200 | Accepted constrained-Qwen output |
| 12 | 747-300 | Accepted constrained-Qwen output |
| 13 | 747-400 | Accepted constrained-Qwen output |
| 14 | 757-200 | Deterministic hierarchy fallback |
| 15 | 757-300 | Deterministic hierarchy fallback |
| 16 | 767-200 | Accepted constrained-Qwen output |
| 17 | 767-300 | Accepted constrained-Qwen output |
| 18 | 767-400 | Accepted constrained-Qwen output |
| 19 | 777-200 | Accepted constrained-Qwen output |
| 20 | 777-300 | Accepted constrained-Qwen output |
| 21 | A300B4 | Accepted constrained-Qwen output |
| 22 | A310 | Deterministic hierarchy fallback |
| 23 | A318 | Accepted constrained-Qwen output |
| 24 | A319 | Accepted constrained-Qwen output |
| 25 | A320 | Accepted constrained-Qwen output |
| 26 | A321 | Accepted constrained-Qwen output |
| 27 | A330-200 | Deterministic hierarchy fallback |
| 28 | A330-300 | Deterministic hierarchy fallback |
| 29 | A340-200 | Accepted constrained-Qwen output |
| 30 | A340-300 | Accepted constrained-Qwen output |
| 31 | A340-500 | Accepted constrained-Qwen output |
| 32 | A340-600 | Accepted constrained-Qwen output |
| 33 | A380 | Accepted constrained-Qwen output |
| 34 | ATR-42 | Deterministic hierarchy fallback |
| 35 | ATR-72 | Deterministic hierarchy fallback |
| 36 | An-12 | Deterministic hierarchy fallback |
| 37 | BAE 146-200 | Accepted constrained-Qwen output |
| 38 | BAE 146-300 | Accepted constrained-Qwen output |
| 39 | BAE-125 | Deterministic hierarchy fallback |
| 40 | Beechcraft 1900 | Deterministic hierarchy fallback |
| 41 | Boeing 717 | Deterministic hierarchy fallback |
| 42 | C-130 | Accepted constrained-Qwen output |
| 43 | C-47 | Deterministic hierarchy fallback |
| 44 | CRJ-200 | Deterministic hierarchy fallback |
| 45 | CRJ-700 | Accepted constrained-Qwen output |
| 46 | CRJ-900 | Deterministic hierarchy fallback |
| 47 | Cessna 172 | Deterministic hierarchy fallback |
| 48 | Cessna 208 | Deterministic hierarchy fallback |
| 49 | Cessna 525 | Deterministic hierarchy fallback |
| 50 | Cessna 560 | Deterministic hierarchy fallback |
| 51 | Challenger 600 | Deterministic hierarchy fallback |
| 52 | DC-10 | Deterministic hierarchy fallback |
| 53 | DC-3 | Deterministic hierarchy fallback |
| 54 | DC-6 | Deterministic hierarchy fallback |
| 55 | DC-8 | Deterministic hierarchy fallback |
| 56 | DC-9-30 | Accepted constrained-Qwen output |
| 57 | DH-82 | Deterministic hierarchy fallback |
| 58 | DHC-1 | Deterministic hierarchy fallback |
| 59 | DHC-6 | Deterministic hierarchy fallback |
| 60 | DHC-8-100 | Deterministic hierarchy fallback |
| 61 | DHC-8-300 | Deterministic hierarchy fallback |
| 62 | DR-400 | Deterministic hierarchy fallback |
| 63 | Dornier 328 | Deterministic hierarchy fallback |
| 64 | E-170 | Accepted constrained-Qwen output |
| 65 | E-190 | Accepted constrained-Qwen output |
| 66 | E-195 | Accepted constrained-Qwen output |
| 67 | EMB-120 | Deterministic hierarchy fallback |
| 68 | ERJ 135 | Accepted constrained-Qwen output |
| 69 | ERJ 145 | Deterministic hierarchy fallback |
| 70 | Embraer Legacy 600 | Deterministic hierarchy fallback |
| 71 | Eurofighter Typhoon | Deterministic hierarchy fallback |
| 72 | F-16A/B | Accepted constrained-Qwen output |
| 73 | F/A-18 | Deterministic hierarchy fallback |
| 74 | Falcon 2000 | Deterministic hierarchy fallback |
| 75 | Falcon 900 | Deterministic hierarchy fallback |
| 76 | Fokker 100 | Deterministic hierarchy fallback |
| 77 | Fokker 50 | Deterministic hierarchy fallback |
| 78 | Fokker 70 | Deterministic hierarchy fallback |
| 79 | Global Express | Deterministic hierarchy fallback |
| 80 | Gulfstream IV | Accepted constrained-Qwen output |
| 81 | Gulfstream V | Accepted constrained-Qwen output |
| 82 | Hawk T1 | Deterministic hierarchy fallback |
| 83 | Il-76 | Deterministic hierarchy fallback |
| 84 | L-1011 | Deterministic hierarchy fallback |
| 85 | MD-11 | Deterministic hierarchy fallback |
| 86 | MD-80 | Accepted constrained-Qwen output |
| 87 | MD-87 | Accepted constrained-Qwen output |
| 88 | MD-90 | Deterministic hierarchy fallback |
| 89 | Metroliner | Deterministic hierarchy fallback |
| 90 | Model B200 | Deterministic hierarchy fallback |
| 91 | PA-28 | Deterministic hierarchy fallback |
| 92 | SR-20 | Deterministic hierarchy fallback |
| 93 | Saab 2000 | Deterministic hierarchy fallback |
| 94 | Saab 340 | Deterministic hierarchy fallback |
| 95 | Spitfire | Deterministic hierarchy fallback |
| 96 | Tornado | Deterministic hierarchy fallback |
| 97 | Tu-134 | Deterministic hierarchy fallback |
| 98 | Tu-154 | Deterministic hierarchy fallback |
| 99 | Yak-42 | Deterministic hierarchy fallback |

## Quality-screening scope

Semantic quality records use deterministic proxies for relevance, accuracy,
completeness and risk. Inputs include class-name/visual-term coverage, format and
fallback records, deduplicated final-prompt counts and keyword warnings in the
original outputs. Higher proxy scores are preferable for the first three dimensions;
risk ranges from 1 (low) to 5 (high). Passing these rules does not establish factual
correctness by human review. No independent author/advisor rating experiment has
been completed. Classification gains and rule-based screening are separate evidence.

The public source comparisons and saved results retain unfavorable outcomes. This
documentation revision does not replace the original prompts or reinterpret them
as expert-verified descriptions.
