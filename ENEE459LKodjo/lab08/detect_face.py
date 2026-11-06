import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    if box_scores.size == 0:
        return box_scores

    scores = box_scores[:, -1]
    boxes = box_scores[:, :4]

    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]

    picked = []

    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)

        if (0 < top_k == len(picked)) or len(indexes) == 1:
            break

        indexes = indexes[:-1]

        current_box = boxes[current]
        remaining_boxes = boxes[indexes]

        max_xy = np.minimum(
            remaining_boxes[:, 2:], current_box[2:]
        )
        min_xy = np.maximum(
            remaining_boxes[:, :2], current_box[:2]
        )

        intersection_wh = np.maximum(max_xy - min_xy, 0)
        intersection = intersection_wh[:, 0] * intersection_wh[:, 1]

        current_wh = np.maximum(
            current_box[2:] - current_box[:2], 0
        )
        current_area = current_wh[0] * current_wh[1]

        remaining_wh = np.maximum(
            remaining_boxes[:, 2:] - remaining_boxes[:, :2], 0
        )
        remaining_area = remaining_wh[:, 0] * remaining_wh[:, 1]

        union = current_area + remaining_area - intersection

        iou = intersection / np.maximum(union, 1e-10)

        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked]



def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size

    resized_image = image.resize((320, 240))
    image_array = np.asarray(resized_image, dtype=np.float32)

    image_array = (image_array - 127.0) / 128.0
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=available_providers
    )

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_array})

    confidences = outputs[0][0]
    boxes = outputs[1][0]

    face_scores = confidences[:, 1]

    mask = face_scores > conf_threshold
    boxes = boxes[mask]
    face_scores = face_scores[mask]

    if len(boxes) > 0:
        box_scores = np.concatenate(
            [boxes, face_scores.reshape(-1, 1)], axis=1
        )
        picked = hard_nms(box_scores, iou_threshold=0.3)
    else:
        picked = np.empty((0, 5))

    draw = ImageDraw.Draw(image)

    for detection in picked:
        x1, y1, x2, y2, score = detection

        x1 = int(np.clip(x1 * original_width, 0, original_width - 1))
        x2 = int(np.clip(x2 * original_width, 0, original_width - 1))
        y1 = int(np.clip(y1 * original_height, 0, original_height - 1))
        y2 = int(np.clip(y2 * original_height, 0, original_height - 1))

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=3
        )
        draw.text(
            (x1, max(0, y1 - 12)),
            f"Face: {score:.2f}",
            fill="red"
        )

    image.save(output_path)
    print(f"Detected {len(picked)} faces")
    print(f"Saved face detection to {output_path}")


# Edge detection

def non_maximum_suppression(magnitude, angle):
    height, width = magnitude.shape
    output = np.zeros((height, width), dtype=np.float32)

    angle = np.rad2deg(angle) % 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            direction = angle[i, j]

            if direction < 22.5 or direction >= 157.5:
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]

            elif direction < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]

            elif direction < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]

            else:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                output[i, j] = magnitude[i, j]

    return output



def hysteresis_thresholding(img, low_thresh, high_thresh):
    strong = 255
    weak = 50

    result = np.zeros(img.shape, dtype=np.uint8)

    strong_mask = img >= high_thresh
    weak_mask = (img >= low_thresh) & (img < high_thresh)

    result[strong_mask] = strong
    result[weak_mask] = weak

    height, width = img.shape

    weak_pixels = np.argwhere(weak_mask)

    for i, j in weak_pixels:
        i_start = max(0, i - 1)
        i_end = min(height, i + 2)
        j_start = max(0, j - 1)
        j_end = min(width, j + 2)

        neighborhood = result[
            i_start:i_end,
            j_start:j_end
        ]

        if np.any(neighborhood == strong):
            result[i, j] = strong
        else:
            result[i, j] = 0

    return result



def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    image = Image.open(image_path).convert("L")
    image_array = np.asarray(image, dtype=np.float32)

    smoothed = gaussian_filter(image_array, sigma=1.4)

    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    Ky = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=np.float32)

    Gx = convolve(smoothed, Kx)
    Gy = convolve(smoothed, Ky)

    magnitude = np.hypot(Gx, Gy)

    max_magnitude = magnitude.max()

    if max_magnitude > 0:
        magnitude = magnitude / max_magnitude * 255.0

    angle = np.arctan2(Gy, Gx)

    suppressed = non_maximum_suppression(magnitude, angle)

    edges = hysteresis_thresholding(
        suppressed, low, high
    )

    output_image = Image.fromarray(edges.astype(np.uint8))
    output_image.save(output_path)

    print(f"Saved Canny edges to {output_path}")


if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")