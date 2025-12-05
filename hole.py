import cv2
import numpy as np
from typing import Dict, List, Tuple


"""
hole.py

Contains two complementary helpers:
- analyze_hole(...) : simple heuristic scoring used by the Streamlit app/tests
- detect_holes_adaptive(...) : image-based adaptive hole detector (returns annotated image and hole list)

The latter is more experimental and depends on OpenCV. Keep both so existing code/tests continue
to work while enabling image-based analysis.
"""


def analyze_hole(trunk_strength: float = 5.0,
                 hole_depth_cm: float = 0.0,
                 wet: bool = False,
                 exposed_crown: bool = False) -> Dict[str, object]:
    """
    Analyze simple hole/trunk/crown observations and return a result dict.

    Inputs:
    - trunk_strength: 0..10 (10 = very strong, 0 = collapsed)
    - hole_depth_cm: depth of the detected hole in centimeters
    - wet: whether the hole area / trunk looks wet
    - exposed_crown: whether the crown is visibly exposed

    Returns a dict with:
    - "score": float between 0 and 1 (higher => higher probability of infestation/weak tree)
    - "weak": bool (True if the tree is judged weak by heuristics)
    - "details": small dict with contributing factors
    """
    # normalize trunk strength
    ts = max(0.0, min(10.0, float(trunk_strength)))
    trunk_factor = 1.0 - (ts / 10.0)  # 0 if very strong, 1 if absent

    # hole factor: deeper hole = more concerning. assume >10cm significant
    hd = max(0.0, float(hole_depth_cm))
    hole_factor = min(1.0, hd / 20.0)  # 0..1 (20cm or more => 1)

    wet_factor = 1.0 if wet else 0.0
    exposed_factor = 1.0 if exposed_crown else 0.0

    # combine simple weighted sum
    score = (0.45 * hole_factor) + (0.35 * trunk_factor) + (0.1 * wet_factor) + (0.1 * exposed_factor)
    score = max(0.0, min(1.0, score))

    # simple boolean weak judgement
    weak = (ts < 4.0) or (hole_factor > 0.5 and wet)

    details = {
        "trunk_strength": ts,
        "trunk_factor": round(trunk_factor, 3),
        "hole_depth_cm": hd,
        "hole_factor": round(hole_factor, 3),
        "wet": wet,
        "exposed_crown": exposed_crown,
    }

    return {"score": round(score, 3), "weak": weak, "details": details}


def detect_holes_adaptive(image_path: str, method: str = 'combined') -> Tuple[np.ndarray, List[Dict], np.ndarray, np.ndarray]:
    """
    Adaptive hole detection that works with different palm tree textures.

    Args:
        image_path: path to image file
        method: 'dark', 'texture', or 'combined'

    Returns:
        result_img, holes_list, combined_mask, debug_img
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Calculate adaptive threshold based on image statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Dynamic threshold: mean - 1.5*std (captures darker anomalies)
    dynamic_threshold = max(20, int(mean_intensity - 1.5 * std_intensity))

    # Method 1: Dark region detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark_mask = cv2.threshold(blurred, dynamic_threshold, 255, cv2.THRESH_BINARY_INV)

    # Method 2: Texture anomaly detection
    # Use local standard deviation to find texture breaks
    kernel_size = 15
    mean_filter = cv2.blur(gray, (kernel_size, kernel_size))
    mean_sq_filter = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
    variance = mean_sq_filter - (mean_filter.astype(np.float32) ** 2)
    variance = np.maximum(variance, 0)  # Avoid negative values
    std_dev = np.sqrt(variance)

    # Holes have low local texture (low std dev) and are dark
    low_texture = std_dev < (np.mean(std_dev) * 0.7)
    dark_regions = gray < (mean_intensity * 0.85)
    texture_mask = (low_texture & dark_regions).astype(np.uint8) * 255

    # Combine masks based on method
    if method == 'dark':
        combined_mask = dark_mask
    elif method == 'texture':
        combined_mask = texture_mask
    else:  # combined
        combined_mask = cv2.bitwise_or(dark_mask, texture_mask)

    # Clean up mask
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holes = []
    result_img = img.copy()
    debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    img_area = h * w
    min_area = img_area * 0.0002  # 0.02% of image
    max_area = img_area * 0.05  # 5% of image

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area or area > max_area:
            continue

        # Calculate metrics
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = float(w_box) / h_box if h_box > 0 else 0

        # Get mean intensity inside contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_inside = cv2.mean(gray, mask=mask)[0]

        # Calculate contrast with surroundings
        dilated_mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
        surrounding_mask = dilated_mask - mask
        mean_surrounding = cv2.mean(gray, mask=surrounding_mask)[0]
        contrast = mean_surrounding - mean_inside

        # Relaxed filters for subtle holes
        is_circular_enough = circularity > 0.25
        is_aspect_ok = 0.3 < aspect_ratio < 3.0
        is_dark_enough = mean_inside < (mean_intensity * 0.9)
        has_contrast = contrast > 5  # At least 5 intensity units darker

        if is_circular_enough and is_aspect_ok and is_dark_enough and has_contrast:
            holes.append({
                'contour': cnt,
                'area': area,
                'center': (x + w_box // 2, y + h_box // 2),
                'circularity': circularity,
                'mean_intensity': mean_inside,
                'contrast': contrast,
                'bbox': (x, y, w_box, h_box)
            })

            # Draw on result
            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)
            cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(result_img, (x + w_box // 2, y + h_box // 2), 4, (0, 0, 255), -1)

            label = f"H{len(holes)}"
            cv2.putText(result_img, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(debug_img, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return result_img, holes, combined_mask, debug_img


def print_results(holes: List[Dict]):
    """Print detection results"""
    print(f"\n{'=' * 60}")
    print(f"HOLES DETECTED: {len(holes)}")
    print(f"{'=' * 60}")

    for i, hole in enumerate(holes, 1):
        print(f"\nHole {i}:")
        print(f"  Area: {hole['area']:.0f} pixels")
        if 'mean_intensity' in hole:
            print(f"  Brightness: {hole['mean_intensity']:.1f}/255")
        if 'contrast' in hole:
            print(f"  Contrast with surroundings: {hole['contrast']:.1f}")
        if 'circularity' in hole:
            print(f"  Circularity: {hole['circularity']:.2f}")

