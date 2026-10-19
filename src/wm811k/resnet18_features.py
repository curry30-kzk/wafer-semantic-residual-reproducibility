"""Frozen ImageNet ResNet18 feature extraction for complete wafer images."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from tqdm import tqdm

from .fewshot_dataset import ImageRecord


MODEL_IDENTIFIER = "torchvision_resnet18_imagenet1k_v1_frozen"
PREPROCESS_IDENTIFIER = "wm811k_rgb224_fullframe_imagenet_normalize_v1"
AIRCRAFT_PREPROCESS_IDENTIFIER = "fgvc_aircraft_banner20_imagenet_centercrop224_v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "ResNet18 baseline requires torch and torchvision. "
            "Install requirements in the selected virtual environment first."
        ) from exc
    return torch, torchvision


def _normalize_dataset_name(dataset: str) -> str:
    normalized = str(dataset).strip().lower().replace("-", "_")
    aliases = {
        "wm811k": "wm811k",
        "wm_811k": "wm811k",
        "fgvc_aircraft": "fgvc_aircraft",
        "aircraft": "fgvc_aircraft",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported ResNet18 preprocessing dataset: {dataset!r}"
        ) from exc


def resnet18_preprocess_metadata(dataset: str = "wm811k") -> dict[str, Any]:
    """Describe the selected ImageNet transform for reproducibility."""

    dataset_name = _normalize_dataset_name(dataset)
    if dataset_name == "fgvc_aircraft":
        return {
            "identifier": AIRCRAFT_PREPROCESS_IDENTIFIER,
            "input_size": 224,
            "remove_bottom_pixels": 20,
            "resize_shorter_edge": 256,
            "interpolation": "bicubic",
            "center_crop": 224,
            "color_mode": "RGB",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        }

    return {
        "identifier": PREPROCESS_IDENTIFIER,
        "input_size": 224,
        "resize": None,
        "crop": None,
        "color_mode": "RGB",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }


def resolve_device(requested: str, cuda_available: bool | None = None) -> str:
    """Resolve ``auto`` to an explicit device and reject unavailable CUDA."""

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if cuda_available is None:
        torch, _ = _require_torch()
        cuda_available = bool(torch.cuda.is_available())
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested but the selected PyTorch environment has no CUDA support"
        )
    return requested


def build_resnet18_transform(
    dataset: str = "wm811k",
) -> Callable[[Any], Any]:
    """Build the selected deterministic ImageNet validation transform."""

    _, torchvision = _require_torch()
    dataset_name = _normalize_dataset_name(dataset)
    transforms: list[Any] = []
    if dataset_name == "fgvc_aircraft":
        from .clip_features import crop_aircraft_banner

        transforms.extend(
            [
                torchvision.transforms.Lambda(crop_aircraft_banner),
                torchvision.transforms.Resize(
                    256,
                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                ),
                torchvision.transforms.CenterCrop(224),
            ]
        )
    transforms.extend(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return torchvision.transforms.Compose(transforms)


def create_resnet18_encoder(device: str) -> Any:
    """Load the frozen torchvision ImageNet-1K V1 ResNet18 encoder."""

    torch, torchvision = _require_torch()
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    encoder = torchvision.models.resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    return encoder.to(device)


def extract_features(
    records: Sequence[ImageRecord],
    *,
    batch_size: int,
    device: str,
    encoder: Any | None = None,
    transform: Callable[[Any], Any] | None = None,
    dataset: str = "wm811k",
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """Extract one float32 feature vector per unique record path in order."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not records:
        raise ValueError("records must not be empty")
    torch, _ = _require_torch()
    resolved_device = resolve_device(device)
    if encoder is None:
        encoder = create_resnet18_encoder(resolved_device)
    else:
        encoder = encoder.to(resolved_device)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        encoder.eval()

    dataset_name = _normalize_dataset_name(dataset)
    use_default_transform = transform is None
    if transform is None:
        transform = build_resnet18_transform(dataset_name)

    relative_paths = [record.relative_path for record in records]
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("records must have unique relative paths before extraction")

    features_by_path: dict[str, np.ndarray] = {}
    starts = range(0, len(records), batch_size)
    iterator = tqdm(
        starts,
        total=(len(records) + batch_size - 1) // batch_size,
        desc="提取 ResNet18 特征",
        unit="batch",
        disable=not show_progress,
    )
    with torch.inference_mode():
        for start in iterator:
            batch_records = records[start : start + batch_size]
            tensors = []
            for record in batch_records:
                from PIL import Image

                with Image.open(record.absolute_path) as image:
                    rgb_image = image.convert("RGB")
                    if (
                        use_default_transform
                        and dataset_name == "wm811k"
                        and rgb_image.size != (224, 224)
                    ):
                        raise ValueError(
                            f"expected a 224x224 prepared image, got {rgb_image.size}: "
                            f"{record.relative_path}"
                        )
                    tensors.append(transform(rgb_image))
            input_batch = torch.stack(tensors, dim=0).to(resolved_device)
            output = encoder(input_batch)
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise ValueError("encoder must return a 2D feature tensor")
            output_array = output.detach().to("cpu", dtype=torch.float32).numpy()
            if output_array.shape[0] != len(batch_records):
                raise ValueError("encoder output batch size does not match input batch size")
            if not np.isfinite(output_array).all():
                raise ValueError("encoder produced non-finite features")
            for record, feature in zip(batch_records, output_array, strict=True):
                features_by_path[record.relative_path] = feature.copy()
    return features_by_path
