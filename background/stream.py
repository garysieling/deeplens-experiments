import cv2
import datetime

camera = cv2.VideoCapture('http://192.168.1.187:8090/feed.h264')
while True:
  (grabbed, frame) = camera.read()

  if not grabbed:
    break
 
  frame = cv2.resize(frame, (int(1920/4), int(1080/4)), cv2.INTER_AREA)

  cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

  cv2.imshow("Feed", frame)

  key = cv2.waitKey(1) & 0xFF

  if key == ord("q"):
    break

camera.release()
cv2.destroyAllWindows()
