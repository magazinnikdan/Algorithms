import cv2
import numpy as np
import light_median_calc as lmc
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time


# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=3 , blockSize=1)

# Parameters for Lucas Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01),
)

color = np.random.randint(0, 255, (100, 3))

cap = cv2.VideoCapture('vid_4.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)
median_x=[]
median_y=[]
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
    scaler = 5
    a_median = scaler*np.array(lmc.median_calc(p0[:, 0, :]/scaler, 1))
    print(p0[:, 0, :])
    median_x.append(a_median[0])
    median_y.append(a_median[1])
    for i in range(len(p0[:,0,0])):
        frame = cv2.circle(frame, (int(p0[i,0,0]), int(p0[i,0,1])), radius=2, color=(0, 0, 255), thickness=-1)
    frame = cv2.circle(frame, (int(a_median[0]), int(a_median[1])), radius=5, color=(0, 255, 0), thickness=-1)

    cv2.imshow('frame', frame)
    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break
# yhat = savgol_filter(median_y, 9, 3) # window size 51, polynomial order 3
# plt.plot(median_x, median_y, 'ro')
# plt.plot(median_x, yhat)
plt.show()