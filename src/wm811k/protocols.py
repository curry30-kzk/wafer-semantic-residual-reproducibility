"""Dataset-level constraints for supported few-shot experiment protocols."""

from __future__ import annotations

from dataclasses import dataclass


WM811K_PROTOCOL_NAME = "wm811k_official_8way_v1"
WM811K_DEDUP_V2_PROTOCOL_NAME = "wm811k_dedup_8way_v2"
FGVC_AIRCRAFT_PROTOCOL_NAME = "fgvc_aircraft_variant_100way_v1"


@dataclass(frozen=True)
class FewShotProtocolSpec:
    """Immutable formal-grid requirements shared by loaders and experiments."""

    name: str
    n_way: int
    image_size: int
    shots: tuple[int, ...] = (1, 5, 10)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("protocol name must not be empty")
        if self.n_way <= 0 or self.image_size <= 0:
            raise ValueError("protocol n_way and image_size must be positive")
        if not self.shots or any(shot <= 0 for shot in self.shots):
            raise ValueError("protocol shots must contain positive integers")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("protocol seeds must contain non-negative integers")
        if len(set(self.shots)) != len(self.shots):
            raise ValueError("protocol shots must be unique")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("protocol seeds must be unique")

    @property
    def formal_task_keys(self) -> frozenset[tuple[int, int]]:
        return frozenset(
            (shot, seed) for shot in self.shots for seed in self.seeds
        )

    def manifest_name(self, shot: int, seed: int) -> str:
        return f"{self.n_way}way_{int(shot)}shot_seed{int(seed)}.json"


EXTENDED_WM_SEEDS = tuple(range(10))


_PROTOCOL_SPECS = {
    WM811K_PROTOCOL_NAME: FewShotProtocolSpec(
        name=WM811K_PROTOCOL_NAME,
        n_way=8,
        image_size=224,
    ),
    WM811K_DEDUP_V2_PROTOCOL_NAME: FewShotProtocolSpec(
        name=WM811K_DEDUP_V2_PROTOCOL_NAME,
        n_way=8,
        image_size=224,
        seeds=EXTENDED_WM_SEEDS,
    ),
    FGVC_AIRCRAFT_PROTOCOL_NAME: FewShotProtocolSpec(
        name=FGVC_AIRCRAFT_PROTOCOL_NAME,
        n_way=100,
        image_size=224,
    ),
}


def get_protocol_spec(name: str) -> FewShotProtocolSpec:
    """Resolve a supported protocol or reject an unregistered experiment."""

    try:
        return _PROTOCOL_SPECS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported few-shot protocol: {name!r}") from exc
