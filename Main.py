import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
import imutils
import time
import cv2
import numpy as np

# Create the main window
main = tk.Tk()
main.title("Object Tracking Using Python")
main.geometry("500x500")

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# Global variables
filename = None
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
"dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "monitor", "water", "stones", "iron gates",
"umbrella", "statue", "trees", "building", "flower", "pen", "computer", "keyboard", "mobile", "lights", "scooty", "flag",
"pot", "bat", "scissor", "pin", "clip", "desk", "notebook", "glass", "boll", "bucket", "clock", "candle", "basket", "apartment", 
"ash", "fire", "bedroom", "bed", "attic", "pencile", "alphabet", "cartoon", "fruits", "apple", "banana", "mango", "watermelon", 
"muskmelon", "orange", "grapes", "strawberry", "pomegranate", "papaya", "litchi", "kiwi", "pineapple", "pear", "custardapple", 
"cherrys", "guava", "peach", "dragonfruit", "plum", "metro train", "pole", "mirror", "sand", "animals", "laptop", "tab", "board", 
"billboard", "screens", "house", "bus stop shelter", "displays", "videowall", "cap", "cup", "wire", "idcard", "Leds", "stars", "globe",
"advertising boards", "banners", "papers", "books", "calculator", "watch", "toys", "kinderjoy chocolate", "shelf"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def uploadVideo():
    global filename
    filename = filedialog.askopenfilename(initialdir="videos")
    pathlabel.config(text=filename)
    text.delete('1.0', tk.END)
    text.insert(tk.END, filename + " loaded\n")
    
    vc = cv2.VideoCapture(filename)
    
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=2000)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                if (confidence * 100) > 50:
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                    text.insert(tk.END, "Object detected in video\n")
        
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    vc.release()
    cv2.destroyAllWindows()

def webcamVideo():
    text.delete('1.0', tk.END)
    webcamera = cv2.VideoCapture(0)
    time.sleep(0.25)
    oldFrame = None

    while True:
        ret, frame = webcamera.read()
        if not ret:
            break
        
        frame = imutils.resize(frame, width=2000)
        (h, w) = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if oldFrame is None:
            oldFrame = gray
            continue
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                if (confidence * 100) > 50:
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                    text.insert(tk.END, "Object detected in webcam\n")
        
        cv2.imshow("Webcam Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    webcamera.release()
    cv2.destroyAllWindows()

def exit():
    main.destroy()

# GUI elements
font = ('elephant', 14, 'bold')
title = tk.Label(main, text='OBJECT TRACKING USING OPEN CV')
title.config(bg='white', fg='black')
title.config(font=font)
title.config(height=3, width=150)
title.place(x=0, y=5)

font1 = ('elephant', 13, 'bold')

uploadButton = tk.Button(main, text="Browse System Videos", command=uploadVideo)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)

pathlabel = tk.Label(main)
pathlabel.config(bg='purple', fg='purple')
pathlabel.config(font=font1)
pathlabel.place(x=460, y=100)

webcamButton = tk.Button(main, text="Start Webcam Video Tracking", command=webcamVideo)
webcamButton.place(x=50, y=150)
webcamButton.config(font=font1)

exitButton = tk.Button(main, text="Exit", command=exit)
exitButton.place(x=50, y=200)
exitButton.config(font=font1)

font1 = ('elephant', 12, 'bold')
text = tk.Text(main, height=20, width=135)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=30, y=250)
text.config(font=font1)

main.config(bg='snow3')
main.config(bg='purple')
main.mainloop()
