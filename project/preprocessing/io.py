import cv2

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Image could not be loaded: {image_path}")
    h, w = image.shape[:2]
    return image, h, w, w / h
