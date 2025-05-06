import onnxruntime as ort
import numpy as np
import supervision as sv
import cv2
from pathlib import Path
from typing import Optional
import os

def box_cxcywh_to_xyxy(boxes):
    """Convert center coordinates (cx, cy, w, h) to (x1, y1, x2, y2)"""
    cx, cy, w, h = np.split(boxes, 4, axis=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.concatenate([x1, y1, x2, y2], axis=-1)


class RfDetr:

    def __init__(self, model_path: str, confidence_threshold: float = 0.45, resolution: int = 336):
        self._session = ort.InferenceSession(model_path)
        self._resolution = resolution
        self._input_name = self._session.get_inputs()[0].name
        self._means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self._confidence_threshold = confidence_threshold

    def __call__(self, *inputs, return_info=False, return_bboxes=False):
        img = inputs[0]
        orig_h, orig_w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = self._preprocess(rgb)
        outputs = self._session.run(None, {self._input_name: input_data})
        res = self._postprocess(outputs, np.array([[orig_h, orig_w]]))  # Исправлено здесь
        return res

    def _preprocess(self, img_array: np.array):
        img_array = cv2.resize(img_array, (self._resolution, self._resolution))
        img_array = np.array(img_array, dtype=np.float32) / 255.0
        normalized = (img_array - self._means) / self._stds
        return normalized.transpose(2, 0, 1)[np.newaxis, ...]

    def _postprocess(self, outputs, target_sizes, num_select=300):
        # Проверяем порядок выходов модели
        out_bbox, out_logits = outputs  # Убедитесь, что порядок правильный

        batch_size = out_logits.shape[0]
        num_classes = out_logits.shape[2]

        prob = 1 / (1 + np.exp(-out_logits))
        
        scores = np.zeros((batch_size, num_select))
        labels = np.zeros((batch_size, num_select), dtype=int)
        topk_boxes = np.zeros((batch_size, num_select), dtype=int)
        
        for i in range(batch_size):
            flat_probs = prob[i].ravel()
            topk_indices = np.argpartition(flat_probs, -num_select)[-num_select:]
            topk_values = flat_probs[topk_indices]

            sorted_indices = np.argsort(-topk_values)
            topk_values = topk_values[sorted_indices]
            topk_indices = topk_indices[sorted_indices]

            scores[i] = topk_values
            labels[i] = topk_indices % num_classes
            topk_boxes[i] = topk_indices // num_classes

        boxes = box_cxcywh_to_xyxy(out_bbox)
        
        selected_boxes = np.take_along_axis(boxes, topk_boxes[..., np.newaxis].repeat(4, axis=-1), axis=1)
        
        img_h = target_sizes[:, 0]
        img_w = target_sizes[:, 1]
        scale_fct = np.stack([img_w, img_h, img_w, img_h], axis=1)
        scaled_boxes = selected_boxes * scale_fct[:, np.newaxis, :]
        
        mask = scores[i] >= self._confidence_threshold

        return sv.Detections(
            xyxy=scaled_boxes[i][mask],
            class_id=labels[i][mask],
            confidence=scores[i][mask]
        )


# -------

def process_frame(
    frame: np.ndarray,
    model: callable,
    labelmap: tuple,
) -> np.ndarray:
    """Обрабатывает и аннотирует один кадр"""
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(smart_position=True)

    #rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    resized_frame = cv2.resize(frame, FRAME_SIZE)
    detections = model(resized_frame)

    # Аннотация кадра
    annotated_frame = box_annotator.annotate(
        resized_frame.copy(), detections=detections
    )
    #annotated_frame = box_annotator.annotate(detections=detections)
    #print(labelmap)
    result_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=[labelmap[class_id - 1] for class_id in detections.class_id]
    )
    #result_frame = label_annotator.annotate(labels=[labelmap[class_id] for class_id in detections.class_id])
    return result_frame

#DEFAULT_LABELMAP = ("barcode", "pallet")
FRAME_SIZE = (432, 768)

def process_images_from_folder(
    labelmap, 
    src_folder: Path,
    #target_folder: Path,
    model_path: Optional[Path] = None,
    image_exts: tuple = ('.jpg')
):
    """
    Обрабатывает все изображения из папки src_folder и сохраняет результаты в target_folder.

    Параметры:
        src_folder: Путь к папке с исходными изображениями
        target_folder: Путь для сохранения обработанных изображений
        model_path: Путь к модели детекции
        labelmap: Карта меток для аннотации
        image_exts: Кортеж расширений изображений для обработки
    """

    #DEFAULT_LABELMAP = ("barcode", "pallet")
    FRAME_SIZE = (432, 768)
    #target_folder = Path(target_folder)
    #target_folder.mkdir(exist_ok=True, parents=True)

    train = Path("train")
    src_folder = os.path.join(src_folder, train)
    src_folder = Path(src_folder)
    # Инициализация модели
    model = RfDetr(str(model_path), resolution=336)

    image_files = [f for f in src_folder.iterdir() if f.suffix.lower() in image_exts]
    #total = len(image_files)
    processed_20 = []
    for idx, image_path in enumerate(image_files[:20], 1):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            continue

        processed = process_frame(img, model, labelmap)
        processed_20.append(processed)
    return processed_20
        #out_path = target_folder / image_path.name
        #cv2.imwrite(str(out_path), processed)
        #print(f"[{idx}/{total}] {image_path.name} -> {out_path.name}", end='\r')

if __name__ == "main":
    process_images_from_folder(
        labelmap = tuple,
        src_folder=Path("input_images"),
        #target_folder=Path("output_images"),
        model_path=Path("inference_model.onnx"),
    )
