import importlib.util
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot


def _load_project_interface():
    repository_root = Path(__file__).resolve().parent.parent
    interface_path = repository_root / "HT-Detector_Peng" / "interface.py"
    if not interface_path.is_file():
        raise RuntimeError(
            "The HT-Detector_Peng interface module was not found: {}".format(
                interface_path
            )
        )

    existing_module = sys.modules.get("interface")
    existing_file = getattr(existing_module, "__file__", None)
    if existing_file:
        try:
            if Path(existing_file).resolve() == interface_path.resolve():
                return existing_module
        except (OSError, RuntimeError):
            pass

    spec = importlib.util.spec_from_file_location("interface", interface_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create an import specification for: {}".format(interface_path)
        )

    module = importlib.util.module_from_spec(spec)
    had_existing_module = "interface" in sys.modules
    sys.modules["interface"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        if had_existing_module:
            sys.modules["interface"] = existing_module
        else:
            sys.modules.pop("interface", None)
        raise RuntimeError(
            "Failed to load HT-Detector_Peng interface module from {}: {}".format(
                interface_path, error
            )
        ) from error
    return module


class YoloDetectionWorker(QObject):
    finished = Signal(object, bool, str)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._weight_path = None

    @Slot(str, str)
    def detect(self, image_path, weight_path):
        try:
            import cv2
            import numpy as np

            _load_project_interface()
            from ultralytics import YOLO

            encoded_path = np.fromfile(image_path, dtype=np.uint8)
            source_image = cv2.imdecode(encoded_path, cv2.IMREAD_COLOR)
            if source_image is None:
                raise ValueError("The selected image could not be read: {}".format(image_path))

            if self._model is None or self._weight_path != weight_path:
                self._model = YOLO(weight_path)
                self._weight_path = weight_path

            results = self._model.predict(
                source=source_image,
                device="cpu",
                conf=0.05,
                save=False,
                verbose=False,
            )
            if not results:
                raise RuntimeError("YOLO returned no result for the selected image.")

            result = results[0]
            annotated_image = source_image.copy()
            boxes = result.boxes
            has_detections = boxes is not None and len(boxes) > 0

            if has_detections:
                coordinates = boxes.xyxy.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                confidences = boxes.conf.detach().cpu().numpy()
                names = result.names

                for coordinates_xyxy, class_id, confidence in zip(
                    coordinates, classes, confidences
                ):
                    x1, y1, x2, y2 = (int(value) for value in coordinates_xyxy)
                    class_name = (
                        names.get(class_id, str(class_id))
                        if isinstance(names, dict)
                        else names[class_id]
                    )
                    label = "{} {:.2f}".format(class_name, confidence)
                    color = (0, 255, 0)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                    )
                    text_top = max(0, y1 - text_height - baseline - 4)
                    cv2.rectangle(
                        annotated_image,
                        (x1, text_top),
                        (x1 + text_width + 4, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        annotated_image,
                        label,
                        (x1 + 2, max(text_height, y1 - baseline - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

            message = (
                "Detection completed."
                if has_detections
                else "Detection completed, but no objects were found. The original image is displayed."
            )
            self.finished.emit(annotated_image, has_detections, message)
        except Exception as error:
            self.failed.emit("{}: {}".format(type(error).__name__, error))
