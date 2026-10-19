"""Frozen OpenAI CLIP ViT-B/32 encoding for complete wafer maps."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from tqdm import tqdm

from .fewshot_dataset import ImageRecord
from .prototype import l2_normalize
from .resnet18_features import resolve_device


MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED_NAME = "openai"
MODEL_IDENTIFIER = "open_clip:ViT-B-32-quickgelu:openai"
PREPROCESS_IDENTIFIER = "wm811k_rgb224_fullframe_openai_clip_normalize_v1"
AIRCRAFT_PREPROCESS_IDENTIFIER = "fgvc_aircraft_banner20_clip_centercrop224_v1"
TIP_ADAPTER_AUGMENTATION_IDENTIFIER = (
    "tip_adapter_random_resized_crop_hflip_clip_normalize_v1"
)
AIRCRAFT_TIP_ADAPTER_AUGMENTATION_IDENTIFIER = (
    "fgvc_aircraft_banner20_tip_adapter_augmentation_v1"
)
PROMPT_TEMPLATE = (
    "a semiconductor wafer map showing {article} {class_name} defect pattern"
)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLASS_TEXT_NAMES = {
    "Center": "center",
    "Donut": "donut",
    "Edge-Loc": "edge-localized",
    "Edge-Ring": "edge-ring",
    "Loc": "localized",
    "Random": "random",
    "Scratch": "scratch",
    "Near-full": "near-full",
}


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CLIP baselines require PyTorch in the selected virtual environment"
        ) from exc
    return torch


def build_class_prompts(class_names: Sequence[str]) -> tuple[str, ...]:
    """Build the frozen generic prompt in explicit manifest class order."""

    names = tuple(str(name) for name in class_names)
    if not names:
        raise ValueError("class_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    prompts = []
    for name in names:
        try:
            text_name = CLASS_TEXT_NAMES[name]
        except KeyError as exc:
            raise ValueError(f"unsupported WM-811K class for CLIP prompt: {name!r}") from exc
        article = "an" if text_name[0].lower() in "aeiou" else "a"
        prompts.append(
            PROMPT_TEMPLATE.format(article=article, class_name=text_name)
        )
    return tuple(prompts)


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
        raise ValueError(f"unsupported CLIP preprocessing dataset: {dataset!r}") from exc


def crop_aircraft_banner(image: Any) -> Any:
    """Remove the official 20-pixel copyright banner from an aircraft image."""

    width, height = image.size
    if height <= 20:
        raise ValueError(
            f"aircraft image height must exceed the 20-pixel banner, got {height}"
        )
    return image.crop((0, 0, width, height - 20))


def clip_preprocess_metadata(dataset: str = "wm811k") -> dict[str, Any]:
    """Describe the selected CLIP input transform for reproducibility."""

    dataset_name = _normalize_dataset_name(dataset)
    if dataset_name == "fgvc_aircraft":
        return {
            "identifier": AIRCRAFT_PREPROCESS_IDENTIFIER,
            "input_size": 224,
            "remove_bottom_pixels": 20,
            "resize_shorter_edge": 224,
            "interpolation": "bicubic",
            "center_crop": 224,
            "color_mode": "RGB",
            "mean": list(CLIP_MEAN),
            "std": list(CLIP_STD),
        }

    return {
        "identifier": PREPROCESS_IDENTIFIER,
        "input_size": 224,
        "resize": None,
        "crop": None,
        "color_mode": "RGB",
        "mean": list(CLIP_MEAN),
        "std": list(CLIP_STD),
    }


def tip_adapter_augmentation_metadata(
    dataset: str = "wm811k",
) -> dict[str, Any]:
    """Describe the support-only transform from the Tip-Adapter reference code."""

    dataset_name = _normalize_dataset_name(dataset)
    if dataset_name == "fgvc_aircraft":
        return {
            "identifier": AIRCRAFT_TIP_ADAPTER_AUGMENTATION_IDENTIFIER,
            "input_size": 224,
            "remove_bottom_pixels": 20,
            "random_resized_crop_scale": [0.5, 1.0],
            "interpolation": "bicubic",
            "horizontal_flip_probability": 0.5,
            "color_mode": "RGB",
            "mean": list(CLIP_MEAN),
            "std": list(CLIP_STD),
        }
    return {
        "identifier": TIP_ADAPTER_AUGMENTATION_IDENTIFIER,
        "input_size": 224,
        "random_resized_crop_scale": [0.5, 1.0],
        "interpolation": "bicubic",
        "horizontal_flip_probability": 0.5,
        "color_mode": "RGB",
        "mean": list(CLIP_MEAN),
        "std": list(CLIP_STD),
    }


def build_clip_transform(dataset: str = "wm811k") -> Callable[[Any], Any]:
    """Build the selected deterministic CLIP validation transform."""

    try:
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "CLIP baselines require torchvision in the selected virtual environment"
        ) from exc
    dataset_name = _normalize_dataset_name(dataset)
    transforms: list[Any] = []
    if dataset_name == "fgvc_aircraft":
        transforms.extend(
            [
                torchvision.transforms.Lambda(crop_aircraft_banner),
                torchvision.transforms.Resize(
                    224,
                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                ),
                torchvision.transforms.CenterCrop(224),
            ]
        )
    transforms.extend(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
    return torchvision.transforms.Compose(transforms)


def build_tip_adapter_augmentation_transform(
    dataset: str = "wm811k",
) -> Callable[[Any], Any]:
    """Build the published Tip-Adapter support augmentation transform."""

    try:
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "Tip-Adapter baselines require torchvision in the selected environment"
        ) from exc
    dataset_name = _normalize_dataset_name(dataset)
    transforms: list[Any] = []
    if dataset_name == "fgvc_aircraft":
        transforms.append(torchvision.transforms.Lambda(crop_aircraft_banner))
    transforms.extend(
        [
            torchvision.transforms.RandomResizedCrop(
                size=224,
                scale=(0.5, 1.0),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
    return torchvision.transforms.Compose(transforms)


class FrozenClipEncoder:
    """One frozen CLIP model shared by image and text evaluation paths."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Callable[[Sequence[str]], Any],
        transform: Callable[[Any], Any],
        augmentation_transform: Callable[[Any], Any] | None = None,
        device: str,
        open_clip_version: str,
        required_image_size: tuple[int, int] | None = (224, 224),
        preprocess_identifier: str = PREPROCESS_IDENTIFIER,
        preprocess_metadata: dict[str, Any] | None = None,
        prompt_builder: Callable[[Sequence[str]], tuple[str, ...]] | None = None,
    ) -> None:
        torch = _require_torch()
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.transform = transform
        self.augmentation_transform = augmentation_transform
        self.open_clip_version = str(open_clip_version)
        self.required_image_size = required_image_size
        self.preprocess_identifier = str(preprocess_identifier)
        self.preprocess_metadata = (
            clip_preprocess_metadata()
            if preprocess_metadata is None
            else dict(preprocess_metadata)
        )
        self.prompt_builder = build_class_prompts if prompt_builder is None else prompt_builder
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self._torch = torch

    @classmethod
    def create(
        cls, device: str, dataset: str = "wm811k"
    ) -> "FrozenClipEncoder":
        """Load the exact OpenCLIP architecture and public OpenAI weights."""

        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "CLIP baselines require open_clip_torch==3.3.0. "
                "Install the project requirements in the selected virtual environment."
            ) from exc
        try:
            model = open_clip.create_model(MODEL_NAME, pretrained=PRETRAINED_NAME)
            tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load OpenCLIP model {MODEL_NAME!r} with "
                f"pretrained weights {PRETRAINED_NAME!r}"
            ) from exc
        try:
            version = importlib.metadata.version("open_clip_torch")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        dataset_name = _normalize_dataset_name(dataset)
        if dataset_name == "fgvc_aircraft":
            from fgvc_aircraft.prompts import build_aircraft_class_prompts

            prompt_builder = build_aircraft_class_prompts
            required_image_size = None
            preprocess_identifier = AIRCRAFT_PREPROCESS_IDENTIFIER
        else:
            prompt_builder = build_class_prompts
            required_image_size = (224, 224)
            preprocess_identifier = PREPROCESS_IDENTIFIER
        return cls(
            model=model,
            tokenizer=tokenizer,
            transform=build_clip_transform(dataset_name),
            augmentation_transform=build_tip_adapter_augmentation_transform(
                dataset_name
            ),
            device=device,
            open_clip_version=version,
            required_image_size=required_image_size,
            preprocess_identifier=preprocess_identifier,
            preprocess_metadata=clip_preprocess_metadata(dataset_name),
            prompt_builder=prompt_builder,
        )

    def encode_images(
        self,
        records: Sequence[ImageRecord],
        *,
        batch_size: int,
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        """Encode unique prepared images once while preserving record order."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not records:
            raise ValueError("records must not be empty")
        relative_paths = [record.relative_path for record in records]
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("records must have unique relative paths before extraction")

        features_by_path: dict[str, np.ndarray] = {}
        starts = range(0, len(records), batch_size)
        iterator = tqdm(
            starts,
            total=(len(records) + batch_size - 1) // batch_size,
            desc="Extract CLIP image features",
            unit="batch",
            disable=not show_progress,
        )
        with self._torch.inference_mode():
            for start in iterator:
                batch_records = records[start : start + batch_size]
                tensors = []
                for record in batch_records:
                    from PIL import Image

                    with Image.open(record.absolute_path) as image:
                        rgb_image = image.convert("RGB")
                        if (
                            self.required_image_size is not None
                            and rgb_image.size != self.required_image_size
                        ):
                            raise ValueError(
                                f"expected a prepared {self.required_image_size[0]}x"
                                f"{self.required_image_size[1]} image, got "
                                f"{rgb_image.size}: {record.relative_path}"
                            )
                        tensors.append(self.transform(rgb_image))
                input_batch = self._torch.stack(tensors, dim=0).to(self.device)
                output = self.model.encode_image(input_batch)
                if not isinstance(output, self._torch.Tensor) or output.ndim != 2:
                    raise ValueError("CLIP image encoder must return a 2D feature tensor")
                if output.shape[0] != len(batch_records):
                    raise ValueError(
                        "CLIP image encoder output batch size does not match input"
                    )
                output_array = output.detach().to(
                    "cpu", dtype=self._torch.float32
                ).numpy()
                if not np.isfinite(output_array).all():
                    raise ValueError("CLIP image encoder produced non-finite features")
                for record, feature in zip(batch_records, output_array, strict=True):
                    features_by_path[record.relative_path] = feature.copy()
        return features_by_path

    def encode_augmented_images(
        self,
        records: Sequence[ImageRecord],
        *,
        batch_size: int,
        view_count: int,
        seed: int,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode deterministic normalized augmentation views in record order."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if view_count <= 0:
            raise ValueError("view_count must be positive")
        if not records:
            raise ValueError("records must not be empty")
        relative_paths = [record.relative_path for record in records]
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("records must have unique relative paths before extraction")
        if self.augmentation_transform is None:
            raise RuntimeError("encoder has no support augmentation transform")

        batches_per_view = (len(records) + batch_size - 1) // batch_size
        progress = tqdm(
            total=view_count * batches_per_view,
            desc="Extract augmented CLIP support features",
            unit="batch",
            disable=not show_progress,
        )
        view_features: list[np.ndarray] = []
        try:
            with self._torch.inference_mode(), self._torch.random.fork_rng(devices=[]):
                for view_index in range(view_count):
                    self._torch.manual_seed(int(seed) + view_index)
                    current_features = []
                    for start in range(0, len(records), batch_size):
                        batch_records = records[start : start + batch_size]
                        tensors = []
                        for record in batch_records:
                            from PIL import Image

                            with Image.open(record.absolute_path) as image:
                                rgb_image = image.convert("RGB")
                                if (
                                    self.required_image_size is not None
                                    and rgb_image.size != self.required_image_size
                                ):
                                    raise ValueError(
                                        "expected a prepared "
                                        f"{self.required_image_size[0]}x"
                                        f"{self.required_image_size[1]} image, got "
                                        f"{rgb_image.size}: {record.relative_path}"
                                    )
                                tensors.append(self.augmentation_transform(rgb_image))
                        input_batch = self._torch.stack(tensors, dim=0).to(self.device)
                        output = self.model.encode_image(input_batch)
                        if not isinstance(output, self._torch.Tensor) or output.ndim != 2:
                            raise ValueError(
                                "CLIP image encoder must return a 2D feature tensor"
                            )
                        current_features.append(
                            output.detach()
                            .to("cpu", dtype=self._torch.float32)
                            .numpy()
                        )
                        progress.update(1)
                    feature_matrix = np.concatenate(current_features, axis=0)
                    if feature_matrix.shape[0] != len(records):
                        raise ValueError(
                            "augmented feature count does not match support records"
                        )
                    if not np.isfinite(feature_matrix).all():
                        raise ValueError(
                            "CLIP image encoder produced non-finite augmented features"
                        )
                    view_features.append(
                        l2_normalize(feature_matrix).astype(np.float32, copy=False)
                    )
        finally:
            progress.close()
        return np.stack(view_features, axis=0)

    def encode_class_text_prototypes(
        self, class_names: Sequence[str]
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Encode and normalize one fixed generic prompt per class."""

        prompts = self.prompt_builder(class_names)
        return self.encode_text_prototypes(prompts), prompts

    def encode_prompt_groups(
        self, prompt_groups: Sequence[Sequence[str]]
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Mean normalized prompt embeddings within each ordered class."""

        groups = tuple(
            tuple(str(prompt).strip() for prompt in group)
            for group in prompt_groups
        )
        if not groups:
            raise ValueError("prompt_groups must not be empty")
        if any(not group for group in groups):
            raise ValueError("every class needs at least one prompt")
        if any(not prompt for group in groups for prompt in group):
            raise ValueError("prompt groups must not contain blank strings")
        flattened = tuple(prompt for group in groups for prompt in group)
        rows = self.encode_text_prototypes(flattened)
        prototypes = []
        offset = 0
        for group in groups:
            next_offset = offset + len(group)
            class_rows = l2_normalize(rows[offset:next_offset])
            class_mean = class_rows.mean(axis=0, keepdims=True)
            prototypes.append(l2_normalize(class_mean)[0])
            offset = next_offset
        return (
            np.stack(prototypes).astype(np.float32, copy=False),
            flattened,
        )

    def encode_text_prototypes(self, prompts: Sequence[str]) -> np.ndarray:
        """Encode and normalize an explicit, immutable prompt sequence."""

        prompt_tuple = tuple(str(prompt) for prompt in prompts)
        if not prompt_tuple:
            raise ValueError("prompts must not be empty")
        if any(not prompt.strip() for prompt in prompt_tuple):
            raise ValueError("prompts must not contain blank strings")
        if len(set(prompt_tuple)) != len(prompt_tuple):
            raise ValueError("prompts must be unique")
        tokens = self.tokenizer(list(prompt_tuple))
        if not isinstance(tokens, self._torch.Tensor):
            raise ValueError("OpenCLIP tokenizer must return a tensor")
        with self._torch.inference_mode():
            output = self.model.encode_text(tokens.to(self.device))
        if not isinstance(output, self._torch.Tensor) or output.ndim != 2:
            raise ValueError("CLIP text encoder must return a 2D feature tensor")
        if output.shape[0] != len(prompt_tuple):
            raise ValueError("CLIP text encoder row count must match class prompts")
        output_array = output.detach().to(
            "cpu", dtype=self._torch.float32
        ).numpy()
        if not np.isfinite(output_array).all():
            raise ValueError("CLIP text encoder produced non-finite features")
        return l2_normalize(output_array).astype(np.float32, copy=False)

    def runtime_metadata(self, feature_dimension: int) -> dict[str, Any]:
        """Return exact model and software identifiers for result JSON."""

        if feature_dimension <= 0:
            raise ValueError("feature_dimension must be positive")
        device = self._torch.device(self.device)
        if device.type == "cuda":
            device_index = (
                self._torch.cuda.current_device()
                if device.index is None
                else device.index
            )
            device_name = self._torch.cuda.get_device_name(device_index)
        else:
            device_name = "CPU"
        return {
            "model_identifier": MODEL_IDENTIFIER,
            "model_name": MODEL_NAME,
            "pretrained": PRETRAINED_NAME,
            "open_clip_torch_version": self.open_clip_version,
            "torch_version": self._torch.__version__,
            "cuda_version": self._torch.version.cuda,
            "cudnn_version": self._torch.backends.cudnn.version(),
            "device_type": device.type,
            "device_name": device_name,
            "feature_dimension": int(feature_dimension),
            "logit_scale_used": False,
            "preprocess_identifier": self.preprocess_identifier,
            "preprocess": dict(self.preprocess_metadata),
        }
