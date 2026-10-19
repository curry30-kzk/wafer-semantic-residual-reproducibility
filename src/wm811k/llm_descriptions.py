"""Frozen structured LLM-generated semantic prompts for WM-811K."""

from __future__ import annotations

from collections.abc import Sequence

from .labels import DEFECT_CLASSES


LLM_STRUCTURED_PROMPT_VERSION = "wm811k_codex_gpt5_structured_v1"
LLM_STRUCTURED_PROMPT_SOURCE = "openai_codex_gpt5_family_design_session"
LLM_STRUCTURED_PROMPT_PROVENANCE = (
    "Generated once in an OpenAI Codex GPT-5-family design session; exact "
    "prompts frozen before formal evaluation; no Test-based revision."
)

LLM_STRUCTURED_PROMPTS: dict[str, tuple[str, str, str, str]] = {
    "Center": (
        "A semiconductor wafer defect map with defective dies concentrated around the geometric center of the circular wafer.",
        "A semiconductor wafer defect map containing one compact, roughly circular or irregular defect cluster in the middle.",
        "A semiconductor wafer defect map whose central region is defective while most surrounding and edge regions remain clear.",
        "A semiconductor wafer defect map with a centered cluster, not an annular donut, an outer-edge ring, or an off-center local cluster.",
    ),
    "Donut": (
        "A semiconductor wafer defect map with defects located in a band surrounding the center but away from the outer circumference.",
        "A semiconductor wafer defect map whose defective dies form an annular, hollow, donut-shaped structure.",
        "A semiconductor wafer defect map with a relatively clear center and outer area separated by a defective middle-radius band.",
        "A semiconductor wafer defect map with an interior ring around the center, distinct from a solid central cluster or a ring on the wafer edge.",
    ),
    "Edge-Loc": (
        "A semiconductor wafer defect map with defects concentrated at one limited angular section of the outer wafer boundary.",
        "A semiconductor wafer defect map containing a compact arc-shaped or blob-shaped local cluster that touches the edge.",
        "A semiconductor wafer defect map where only a small portion of the circumference is defective and the remaining edge is mostly clear.",
        "A semiconductor wafer defect map with a localized edge cluster rather than a continuous edge ring or an interior local cluster.",
    ),
    "Edge-Ring": (
        "A semiconductor wafer defect map with defective dies distributed along most of the outer circumference of the wafer.",
        "A semiconductor wafer defect map whose defects form a continuous or nearly continuous circular ring at the boundary.",
        "A semiconductor wafer defect map with high defect coverage around the perimeter while much of the inner wafer remains clearer.",
        "A semiconductor wafer defect map with a circumference-wide edge band, distinct from one localized edge patch or an interior donut ring.",
    ),
    "Loc": (
        "A semiconductor wafer defect map with a defect concentration in one off-center interior region that does not touch the boundary.",
        "A semiconductor wafer defect map containing one compact irregular blob or cluster inside the wafer.",
        "A semiconductor wafer defect map where a limited interior area is defective and most other wafer sites remain clear.",
        "A semiconductor wafer defect map with an interior localized cluster, distinct from a centered cluster, an edge-localized cluster, or random scatter.",
    ),
    "Random": (
        "A semiconductor wafer defect map with defective dies appearing at many unrelated locations across the wafer.",
        "A semiconductor wafer defect map containing isolated points and small irregular fragments without a stable geometric shape.",
        "A semiconductor wafer defect map with sparse or moderate defects dispersed broadly rather than concentrated in one region.",
        "A semiconductor wafer defect map showing unstructured scatter, not a compact cluster, line-like scratch, ring, or near-full coverage.",
    ),
    "Scratch": (
        "A semiconductor wafer defect map with defective dies aligned across an elongated path through part of the wafer.",
        "A semiconductor wafer defect map whose defects form a narrow linear, diagonal, curved, or streak-like structure.",
        "A semiconductor wafer defect map with concentrated lengthwise coverage and relatively small width compared with the wafer diameter.",
        "A semiconductor wafer defect map with a coherent line-like streak, distinct from random scattered points or a compact local blob.",
    ),
    "Near-full": (
        "A semiconductor wafer defect map with defective dies occupying almost all interior locations across the wafer.",
        "A semiconductor wafer defect map forming a dense wafer-shaped mass rather than a localized geometric pattern.",
        "A semiconductor wafer defect map with very high defect coverage and only a small number of clear sites remaining.",
        "A semiconductor wafer defect map dominated by near-complete coverage, distinct from sparse random defects or any localized cluster or ring.",
    ),
}


def build_llm_structured_prompt_ensemble(
    class_names: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    """Return four frozen prompts per class in the official manifest order."""

    names = tuple(str(name) for name in class_names)
    if names != DEFECT_CLASSES:
        raise ValueError(
            "structured LLM prompts require the official 8-class WM-811K order"
        )
    return tuple(LLM_STRUCTURED_PROMPTS[name] for name in names)
