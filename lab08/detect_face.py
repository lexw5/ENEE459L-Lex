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
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []

    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]

    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)

        if len(picked) == top_k or len(indexes) == 1:
            break

        current_box = boxes[current]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes]
        
        x1_max = np.maximum(rest_boxes[:, 0], current_box[0])
        y1_max = np.maximum(rest_boxes[:, 1], current_box[1])
        x2_min = np.minimum(rest_boxes[:, 2], current_box[2])
        y2_min = np.minimum(rest_boxes[:, 3], current_box[3])

        intersection = np.maximum(0.0, x2_min - x1_max) * np.maximum(0.0, y2_min - y1_max)

        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_rest = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        union = area_current + area_rest - intersection

        iou = intersection/union

        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    resized = image.resize((320, 240))
    img = np.asarray(resized).astype(np.float32) 
    img = (img - 127.0) / 128.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=available_providers)
    input_name = session.get_inputs()[0].name
    scores, boxes = session.run(None, {input_name: img})

    scores = scores[0]
    boxes = boxes[0]
    probs = scores[:, 1]

    mask = probs >= conf_threshold
    probs = probs[mask]
    subset_boxes = boxes[mask]

    box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
    box_probs = hard_nms(box_probs, iou_threshold=0.3)

    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, prob in box_probs:
        x1, x2 = x1 * orig_w, x2 * orig_w
        y1, y2 = y1 * orig_h, y2 * orig_h
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1, max(0, y1 - 12)), f"{prob:.2f}", fill=(0, 255, 0))

    image.save(output_path)
    return box_probs

# Edge detection
def non_maximum_suppression(magnitude, angle):
    H, W = magnitude.shape
    output = np.zeros((H, W), dtype = np.float32)

    angle = np.rad2deg(angle) % 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            theta = angle[i, j]

            if (0 <= theta < 22.5) or (157.5 <= theta <= 180):
                q = magnitude[i, j - 1]
                r = magnitude[i, j + 1]
            elif (22.5 <= theta < 67.5):
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= theta < 112.5:
                q = magnitude[i - 1, j]
                r = magnitude[i + 1, j]
            else:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                output[i, j] = magnitude[i, j]

    return output

def hysteresis_thresholding(img, low_thresh, high_thresh):
    STRONG, WEAK = 255, 60

    result = np.zeros(img.shape, dtype=np.uint8)
    strong_mask = img >= high_thresh
    weak_mask = (img >= low_thresh) & (img <= high_thresh)
    result[strong_mask] = STRONG
    result[weak_mask] = WEAK

    weak_i, weak_j = np.where(result == WEAK)
    for i, j in zip(weak_i, weak_j):
        window = result[max(i-1, 0):i+2, max(j-1, 0):j+2]
        result[i, j] = STRONG if np.any(window == STRONG) else 0

    return result

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    gray = Image.open(image_path).convert("L")
    gray = np.asarray(gray).astype(np.float32)
    smoothed = gaussian_filter(gray, sigma = 1.4)

    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)

    Gx = convolve(smoothed, Kx)
    Gy = convolve(smoothed, Ky)

    magnitude = np.hypot(Gx, Gy)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255.0
    angle = np.arctan2(Gy, Gx)

    thinned = non_maximum_suppression(magnitude, angle)

    edges = hysteresis_thresholding(thinned, low, high)

    Image.fromarray(edges).save(output_path)
    return edges



if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")
