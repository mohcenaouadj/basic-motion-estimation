import cv2
import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('TkAgg')

cap = cv2.VideoCapture('shibuya.mp4')
color = (0, 255, 0)
feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7) # Shi-Tomasi parameters
lk_params = dict(winSize = (15, 150), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) # Lucas-Kanade parameters
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
mask = np.zeros_like(first_frame)


fig, ax = plt.subplots(1, 2)
ax[0].imshow(first_frame)
ax[1].imshow(prev_gray, cmap = 'gray')
plt.show()

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    good_old = prev[status == 1]
    good_new = next[status == 1]
    prev = good_new.reshape(-1, 1, 2)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
        frame = cv2.circle(frame, (int(a), int(b)),3, color, 2)
    prev_gray = gray.copy()
    output = cv2.add(frame, mask)
    cv2.imshow("Sparse optical flow", output)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

