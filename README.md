**Driver Drowsiness Detection**

**Technologies** 
Flask · OpenCV · Keras (CNN)

 **Overview**

This project aims to enhance road safety by detecting driver drowsiness in real time.
Using a webcam, the system monitors the driver’s eyes and uses a Convolutional Neural Network (CNN) to determine whether they are open or closed.
If the eyes remain closed for a certain duration, an audible alarm is triggered to alert the driver.

**Features**

Real-time video stream analysis using OpenCV

Eye state detection with a trained CNN model (Keras)

Flask web interface for launching and controlling the system

Automatic alarm trigger when drowsiness is detected

**Tech Stack**

Python

Flask (for web interface)

OpenCV (for real-time image processing)

Keras / TensorFlow (for CNN model)

NumPy, imutils, Pygame (for supporting functions and audio)
