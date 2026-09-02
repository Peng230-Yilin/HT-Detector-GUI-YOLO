"""Pure batch, pairing, spatial ordering, and numbering primitives."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import os
from pathlib import Path
import re
from statistics import median
from typing import Iterable, List, Optional, Sequence, Tuple


Box = Tuple[float, float, float, float]
_NATURAL_PART = re.compile(r"(\d+)")


class DetectionScope(str, Enum):
    CURRENT_IMAGE = "current_image"
    ALL_IMPORTED_IMAGES = "all_imported_images"


class NumberingMode(str, Enum):
    PER_IMAGE = "per_image"
    CONTINUOUS_BATCH = "continuous_batch"


class ImageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SampleStatus(str, Enum):
    VALID = "valid"


class SampleErrorType(str, Enum):
    UNMATCHED_CUVETTE = "unmatched_cuvette"
    UNMATCHED_LIQUID = "unmatched_liquid"
    AMBIGUOUS_CUVETTE = "ambiguous_cuvette"
    AMBIGUOUS_LIQUID = "ambiguous_liquid"
    INVALID_ROI = "invalid_roi"
    MEASUREMENT_FAILED = "measurement_failed"
    IMAGE_FAILED = "image_failed"


@dataclass
class SampleResult:
    image_order: int
    source_file: str
    cuvette_box: Box
    liquid_box: Box
    roi_box: Box
    red: float
    green: float
    blue: float
    con_r: Optional[float] = None
    con_g: Optional[float] = None
    con_b: Optional[float] = None
    no_in_image: Optional[int] = None
    batch_no: Optional[int] = None
    status: SampleStatus = SampleStatus.VALID
    warnings: List[str] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)

    def legacy_target(self, concentration):
        return {
            "No.": self.no_in_image,
            "Con.": concentration,
            "Red": self.red,
            "Green": self.green,
            "Blue": self.blue,
            "cuvette_box": self.cuvette_box,
            "liquid_box": self.liquid_box,
            "rgb_roi": self.roi_box,
        }


@dataclass
class SampleError:
    image_order: int
    source_file: str
    error_type: SampleErrorType
    reason: str
    related_boxes: List[Box] = field(default_factory=list)
    related_cuvette_boxes: List[Box] = field(default_factory=list)
    related_liquid_boxes: List[Box] = field(default_factory=list)
    position: Optional[Tuple[float, float]] = None
    no_in_image: None = field(default=None, init=False)
    batch_no: None = field(default=None, init=False)

    def as_dict(self):
        return asdict(self)


@dataclass
class ImageItem:
    path: str
    original_filename: str
    image_order: int
    status: ImageStatus = ImageStatus.PENDING
    samples: List[SampleResult] = field(default_factory=list)
    errors: List[SampleError] = field(default_factory=list)


_DEFAULT_INDEX = object()


@dataclass(init=False)
class BatchState:
    images: List[ImageItem]
    _current_image_index: Optional[int]
    last_batch_result: Optional[object]
    detection_scope: DetectionScope
    numbering_mode: NumberingMode

    def __init__(
        self,
        images=None,
        current_image_index=_DEFAULT_INDEX,
        last_batch_result=None,
        detection_scope=DetectionScope.CURRENT_IMAGE,
        numbering_mode=NumberingMode.PER_IMAGE,
    ):
        self.images = list(images) if images is not None else []
        self._current_image_index = None
        self.last_batch_result = last_batch_result
        self.detection_scope = detection_scope
        self.numbering_mode = numbering_mode
        if current_image_index is _DEFAULT_INDEX:
            current_image_index = 0 if self.images else None
        self.current_image_index = current_image_index

    @property
    def current_image_index(self) -> Optional[int]:
        self._validate_current_image_index(self._current_image_index)
        return self._current_image_index

    @current_image_index.setter
    def current_image_index(self, value):
        self._validate_current_image_index(value)
        self._current_image_index = value

    def _validate_current_image_index(self, value):
        if not self.images:
            if value is not None:
                raise IndexError("An empty batch has no current image index.")
            return
        if (isinstance(value, bool) or not isinstance(value, int)
                or value < 0 or value >= len(self.images)):
            raise IndexError("Current image index is outside the batch.")

    @property
    def current_image(self) -> Optional[ImageItem]:
        index = self.current_image_index
        return None if index is None else self.images[index]

    @classmethod
    def from_paths(cls, paths: Iterable[os.PathLike]):
        return cls(images=build_image_items(paths))


@dataclass
class PairingResult:
    pairs: List[Tuple[Box, Box]] = field(default_factory=list)
    errors: List[SampleError] = field(default_factory=list)


def normalize_internal_path(path: os.PathLike) -> str:
    """Normalize a path lexically without requiring it to exist."""
    return os.path.normpath(os.path.abspath(os.fspath(path)))


def _natural_text_key(text: str):
    return tuple(
        (0, int(part), len(part)) if part.isdigit() else (1, part.casefold())
        for part in _NATURAL_PART.split(text)
        if part
    )


def natural_sort_paths(paths: Iterable[os.PathLike]) -> List[os.PathLike]:
    """Return inputs in deterministic natural filename order, preserving values."""
    indexed = list(enumerate(paths))

    def key(item):
        index, original = item
        raw = os.fspath(original)
        normalized = normalize_internal_path(raw)
        return (
            _natural_text_key(Path(raw).name),
            normalized.casefold(),
            normalized,
            index,
        )

    return [original for _, original in sorted(indexed, key=key)]


def build_image_items(paths: Iterable[os.PathLike]) -> List[ImageItem]:
    return [
        ImageItem(
            path=normalize_internal_path(path),
            original_filename=Path(os.fspath(path)).name,
            image_order=image_order,
        )
        for image_order, path in enumerate(natural_sort_paths(paths), start=1)
    ]


def _center(box: Box) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _intersection_width(first: Box, second: Box) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0]))


def _candidate_indexes(liquid: Box, cuvettes: Sequence[Box]) -> List[int]:
    center_x, center_y = _center(liquid)
    contained = [
        index
        for index, box in enumerate(cuvettes)
        if box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]
    ]
    if contained:
        return contained
    return [
        index for index, box in enumerate(cuvettes)
        if _intersection_width(liquid, box) > 0.0
    ]


def pair_cuvettes_and_liquids(
    cuvettes: Sequence[Box], liquids: Sequence[Box], image_order=1, source_file=""
) -> PairingResult:
    """Pair only mutually unique geometry candidates and record every rejection."""
    cuvettes = sorted(tuple(map(float, box)) for box in cuvettes)
    liquids = sorted(tuple(map(float, box)) for box in liquids)
    liquid_candidates = [_candidate_indexes(liquid, cuvettes) for liquid in liquids]
    cuvette_candidates = [
        [liquid_index for liquid_index, candidates in enumerate(liquid_candidates)
         if cuvette_index in candidates]
        for cuvette_index in range(len(cuvettes))
    ]
    result = PairingResult()
    paired_cuvettes = set()
    paired_liquids = set()

    for liquid_index, candidates in enumerate(liquid_candidates):
        if len(candidates) != 1:
            continue
        cuvette_index = candidates[0]
        if len(cuvette_candidates[cuvette_index]) == 1:
            result.pairs.append((cuvettes[cuvette_index], liquids[liquid_index]))
            paired_cuvettes.add(cuvette_index)
            paired_liquids.add(liquid_index)

    for index, liquid in enumerate(liquids):
        if index in paired_liquids:
            continue
        candidates = liquid_candidates[index]
        error_type = (SampleErrorType.UNMATCHED_LIQUID if not candidates
                      else SampleErrorType.AMBIGUOUS_LIQUID)
        reason = ("No cuvette candidate for liquid." if not candidates
                  else "Liquid has multiple candidates or competes for a cuvette.")
        result.errors.append(SampleError(
            image_order, source_file, error_type, reason,
            [liquid] + sorted(cuvettes[i] for i in candidates),
            related_cuvette_boxes=sorted(cuvettes[i] for i in candidates),
            related_liquid_boxes=[liquid],
            position=_center(liquid),
        ))

    for index, cuvette in enumerate(cuvettes):
        if index in paired_cuvettes:
            continue
        candidates = cuvette_candidates[index]
        error_type = (SampleErrorType.UNMATCHED_CUVETTE if not candidates
                      else SampleErrorType.AMBIGUOUS_CUVETTE)
        reason = ("No liquid candidate for cuvette." if not candidates
                  else "Cuvette has multiple candidates or is part of an ambiguity.")
        result.errors.append(SampleError(
            image_order, source_file, error_type, reason,
            [cuvette] + sorted(liquids[i] for i in candidates),
            related_cuvette_boxes=[cuvette],
            related_liquid_boxes=sorted(liquids[i] for i in candidates),
            position=_center(cuvette),
        ))

    result.pairs = sort_spatially(result.pairs, box_getter=lambda pair: pair[0])
    result.errors.sort(key=lambda error: (
        error.error_type.value,
        error.position[1] if error.position else float("inf"),
        error.position[0] if error.position else float("inf"),
        tuple(error.related_cuvette_boxes),
        tuple(error.related_liquid_boxes),
        error.reason,
    ))
    return result


def sort_spatially(items: Sequence, box_getter=lambda item: item.cuvette_box) -> List:
    """Sort adaptive rows top-to-bottom and each row left-to-right."""
    if not items:
        return []
    decorated = [(item, tuple(map(float, box_getter(item)))) for item in items]
    typical_height = median(max(0.0, box[3] - box[1]) for _, box in decorated)
    row_tolerance = typical_height * 0.5
    decorated.sort(key=lambda pair: (_center(pair[1])[1], _center(pair[1])[0], pair[1]))
    rows = []
    for item, box in decorated:
        center_x, center_y = _center(box)
        if not rows or center_y - rows[-1]["anchor_y"] > row_tolerance:
            rows.append({"anchor_y": center_y, "values": [(center_x, center_y, box, item)]})
        else:
            rows[-1]["values"].append((center_x, center_y, box, item))
    ordered = []
    for row in rows:
        row["values"].sort(key=lambda value: (value[0], value[1], value[2]))
        ordered.extend(value[-1] for value in row["values"])
    return ordered


def assign_image_numbers(samples: Sequence[SampleResult]) -> List[SampleResult]:
    ordered = sort_spatially(samples)
    for number, sample in enumerate(ordered, start=1):
        sample.no_in_image = number
    return ordered


def assign_batch_numbers(images: Sequence[ImageItem]) -> int:
    for image in images:
        for sample in image.samples:
            sample.batch_no = None
    batch_no = 0
    for image in sorted(images, key=lambda item: item.image_order):
        if image.status != ImageStatus.COMPLETED:
            continue
        image.samples = assign_image_numbers(image.samples)
        for sample in image.samples:
            batch_no += 1
            sample.batch_no = batch_no
    return batch_no
