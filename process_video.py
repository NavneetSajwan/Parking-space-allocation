import cv2

cap = cv2.VideoCapture("sample.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
print(frame_width/2, frame_height/2)
out = cv2.VideoWriter('sample_lr_6.mp4',fourcc, 20.0, (int(frame_width/6), int(frame_height/6)))
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret==True:
	  new_frame = cv2.resize(frame,(int(frame_width/6), int(frame_height/6)),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
	  cv2.imshow('frame', new_frame)
	  out.write(new_frame)
	  if cv2.waitKey(1) & 0xFF == ord('q'):
	      break
	else:
	    break

cap.release()
out.release()
cv2.destroyAllWindows()