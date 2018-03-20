#!/usr/bin/python -tt
# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import smtplib

from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth

if __name__ == '__main__':
	gauth = GoogleAuth()
	drive = GoogleDrive(gauth)
	k=0

	server = smtplib.SMTP('smtp.gmail.com',587)
	server.starttls()
	server.login("sparshbbhs@gmail.com","thisisanewpass")
	msg = "Motion Detected! See Google Drive."

	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=True,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-c", "--confidence", type=float, default=0.0,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	fps = FPS().start()

	while True:
		global frame
		m=0
		count = 0
		frame = vs.read()
		frame = imutils.resize(frame, width=800)

		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)

		net.setInput(blob)
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				label = "{}: {:.2f}%".format(CLASSES[idx],
					confidence * 100)
				if CLASSES[idx]:
					try:
						#print(CLASSES[idx])
						pass
					except IOError:
						print("IOError")

				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				if(CLASSES[idx]=="person"):
					count = count + 1

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

		fps.update()

		if(count>=2):
			print("dead")
			m=m+1

		if(m!=0):
			cv2.imwrite("C:/SavedPictures/frame%d.jpg" % k, frame)
			file2=drive.CreateFile({'parent':'C:/SavedPictures/StoredImages/'})
			file2.SetContentFile('C:/SavedPictures/frame%d.jpg' % k)
			file2.Upload()
			server.sendmail("sparshbbhs@gmail.com","skkumarsparsh@gmail.com",msg)
			m=0

	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	cv2.destroyAllWindows()
	vs.stop()