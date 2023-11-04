import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

image_path = "cat.jpeg"
image = Image.open(image_path)
image_tensor = F.to_tensor(image).unsqueeze(0)

with torch.no_grad():
    prediction = model(image_tensor)

draw = ImageDraw.Draw(image)
for element in range(len(prediction[0]['boxes'])):
    #print(prediction)
    box = prediction[0]['boxes'][element].cpu().numpy()
    score = prediction[0]['scores'][element].cpu().numpy()
    label = COCO_INSTANCE_CATEGORY_NAMES[prediction[0]['labels'][element]]

    #print(f'{label} detected with center at')

    if label in ['cat', 'apple', 'clock'] and score > 0.8: 
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
        draw.ellipse((x_center - 5, y_center - 5, x_center + 5, y_center + 5), fill='blue')
        
        print(f'bounding box midpoint x,y: {x_center}, {y_center}')
        #print(f'{label} detected with center at ({x_center}, {y_center}), with score: {score}')

output_path = "output_image.jpg"
image.save(output_path)
print(f"Output image saved to {output_path}")
