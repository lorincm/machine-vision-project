from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib.pyplot as plt


#load large oiv7 model
model = YOLO('./yolov8x-oiv7.pt')

#results = model('https://www.americanrifleman.org/media/uman22jj/fn-america-high-power-stainless-steel-f.jpg')
results = model('./testimg3.jpg')


detection_result = []

for r in results:    
    boxes = r.boxes.xywh #.numpy()

    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #im.show()  # show image
    im.save('results.jpg')  # save image
    image = cv2.imread('./results.jpg')

    boxes = r.boxes.cpu().numpy()
    #draw image
    for box in boxes:
            xy = box.xywh                    
            x = int(xy[0][0])
            y = int(xy[0][1])

            
            image = cv2.circle(image, (x,y), radius=10, color=(0, 0, 255), thickness=-1)
            cv2.imwrite('results.jpg', image)
            detection_result.append([model.names[box.cls[0]], x, y])
    

print(detection_result)

image = cv2.imread('./results.jpg')
cv2.imshow('image',image)
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows()
