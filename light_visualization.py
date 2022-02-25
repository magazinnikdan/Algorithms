import cv2
import numpy as np
#Range of input is -200 to 200 for both axis
a = [-0, 100]



height = 800
width = 800
s = 700
k = 300
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2

image = np.zeros((height,width,3), np.uint8)
image.fill(255)
start_point_s = [int(height/2-s/2), int(width/2-s/2)]
end_point_s = [int(height/2+s/2), int(width/2+s/2)]
color = (255, 0, 0)
thickness = 2
image = cv2.rectangle(image, start_point_s, end_point_s, color, thickness)
image = cv2.circle(image, [int(height/2), int(height/2)],2,  (0, 255, 0), 1)
#image = cv2.putText(image, '0', [int(height/2)+20, int(height/2)], font, 0.3, color, 1, cv2.LINE_AA)



start_point_s[1]=start_point_s[1]+50
start_point_s[0]=start_point_s[0]+20
image = cv2.putText(image, 'D', start_point_s, font, fontScale, color, thickness, cv2.LINE_AA)
cv2.imshow("Blank", image)



start_point_k = [int(height/2-k/2)+a[0], int(width/2-k/2)-a[1]]
end_point_k = [int(height/2+k/2)+a[0], int(width/2+k/2)-a[1]]
color = (0, 0, 255)
thickness = 2
image = cv2.rectangle(image, start_point_k, end_point_k, color, thickness)
start_point_k[1]=start_point_k[1]+50
start_point_k[0]=start_point_k[0]+20
image = cv2.putText(image, 'D_ell', start_point_k, font, fontScale, color, thickness, cv2.LINE_AA)
cv2.imshow("Blank", image)
cv2.waitKey(0)