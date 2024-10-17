from ultralytics import YOLO
import cv2
import random
import numpy as np
import os



def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    # color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined
            

def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    # color = (0,0,250)

    # if label == "crocodile":
    #     color = (0,250,0)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



# Load a model
model = YOLO("best.pt")
class_names = model.names
print('Class Names: ', class_names)

# colors = (0,0,250)

# if class_names == "crocodile":
#     colors = (0,250,0)



colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]
# cap = cv2.VideoCapture('test.mp4')
output_dir = 'runs/output'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture("boya.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = os.path.join(output_dir, 'output.mp4')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

head_class_index = 1

while True:
    success, img = cap.read()
    if not success:
        break

    h, w, _ = img.shape
    results = model.predict(img, stream=True)
    # print(results)
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            if int(box.cls) == head_class_index:

                seg = cv2.resize(seg, (w, h))
                img = overlay(img, seg, colors[int(box.cls)], 0.4)
                
                xmin = int(box.data[0][0])
                ymin = int(box.data[0][1])
                xmax = int(box.data[0][2])
                ymax = int(box.data[0][3])

                width_pixels = xmax -xmin
                height_pixels = ymax -ymin

                print(f'Width in pixels: {width_pixels}')
                print(f'Height in pixels: {height_pixels}')
                
                plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(box.cls)], f'{class_names[int(box.cls)]} {float(box.conf):.3}')

    out.write(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
