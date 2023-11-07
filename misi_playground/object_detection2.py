from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib.pyplot as plt


#load large oiv7 model
model = YOLO('./yolov8x-oiv7.pt')

#results = model('https://www.americanrifleman.org/media/uman22jj/fn-america-high-power-stainless-steel-f.jpg')
results = model('./testimg2.jpg')



for r in results:    
    boxes = r.boxes.xywh.numpy()

    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #im.show()  # show image
    im.save('results.jpg')  # save image
    image = cv2.imread('./results.jpg')

    #draw image
    for box in boxes:
        x = int(box[0])
        y = int(box[1])
        #print(type(x))
        #print(str(x)+" "+str(y))
        image = cv2.circle(image, (x,y), radius=10, color=(0, 0, 255), thickness=-1)
        cv2.imwrite('results.jpg', image)

image = cv2.imread('./results.jpg')
cv2.imshow('image',image)
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows()
