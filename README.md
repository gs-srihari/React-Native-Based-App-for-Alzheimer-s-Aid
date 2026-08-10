# Real-Time Object Detection and Search using YOLO

A real-time computer vision application that uses **YOLOv2** to detect objects from a live camera feed and provides an object-search interface through a **React Native** mobile application.

> **Project Context**
>
> This project was developed in **2019** as an undergraduate student course project at **Vellore Institute of Technology, Chennai Campus**. The implementation reflects the tools and technologies available and commonly used at the time, including YOLOv2, Darkflow, React Native, and Firebase.

The system records objects detected together in the camera view and stores the detection results in **Firebase Realtime Database**. A user can then search for a particular object and retrieve other objects that were detected alongside it.

## Overview

The project consists of three main components:

1. **Real-Time Object Detection** — YOLOv2 detects objects from a webcam feed.
2. **Cloud Data Storage** — Detected objects are grouped by detection instance and stored in Firebase.
3. **Object Search Application** — A React Native application allows users to search for an object and retrieve associated objects from the stored detection data.

### Example

If the camera detects:

```text
Person
Chair
Bottle
Table
```

the system stores these objects as a group.

Searching for:

```text
Chair
```

can return the other objects detected in the same group:

```text
1. Person
2. Bottle
3. Table
```

## Object Detection Pipeline

The Python application captures frames from a webcam and passes each frame to YOLOv2 for inference.

For every detected object, the system extracts:

- Object class
- Confidence score
- Bounding-box coordinates

The detected objects from each frame are grouped together and stored in Firebase.

For example:

```text
Detection Record
├── data1: person
├── data2: chair
├── data3: bottle
├── data4: table
└── data5: laptop
```

## Object Search

The React Native application retrieves detection records from Firebase.

When a user searches for an object, the application:

1. Searches the stored detection records.
2. Identifies records containing the requested object.
3. Extracts the other objects in those records.
4. Displays the associated objects to the user.

This provides a simple way of finding objects based on their co-occurrence within the camera view.

## Requirements

### Python

Install the required Python dependencies:

```bash
pip install opencv-python
pip install numpy
```

The project also requires **Darkflow** and its associated YOLOv2 configuration and weights.

### Firebase

Configure the Firebase Realtime Database and update the Firebase connection details in the Python application.

### React Native

Install the required React Native / Expo dependencies for the mobile application.

## Running the Object Detection System

Start the Python detection application:

```bash
python object_detect_firebase.py
```

The application opens the webcam, performs real-time YOLOv2 inference, displays detected objects with bounding boxes, and uploads the detected object groups to Firebase.

Press:

```text
Q
```

to exit the application.

## Limitations

- Object detection is limited to the classes supported by the trained YOLOv2 model.
- The original implementation uses an older Darkflow/YOLOv2 stack.
- The system associates objects based on their occurrence within the same detection record rather than computing physical distance between objects.

## Original Motivation

This project was originally conceptualized as an assistive technology for individuals with Alzheimer's disease, with the goal of helping users identify objects in their surroundings and recall objects that had been encountered together. The current repository focuses on the underlying **real-time object detection and object-search system**, independent of that original application context.


## License

This project is provided for educational and research purposes.