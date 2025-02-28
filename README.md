# Star-detection
This project showcases a hybrid approach to detect and visualize constellations in night-sky images. By combining **YOLOv8** (a state-of-the-art object detection model) with classical **computer vision** techniques, it not only identifies constellations but also pinpoints individual stars to create a complete astronomical mapping pipeline.

## Key Features
- utilizes YOLOv8 to accurately detect constellation bounding boxes in a variety of sky images
- applies *brightness* adjustments, *thresholding*, *morphological operations*, and *watershed* segmentation to isolate and locate individual stars
- once constellations are detected, the pipeline matches each star to its corresponding constellation
- easily tweak the detection parameters (confidence threshold, morphological kernel sizes, etc.) to handle different lighting conditions and image qualities
- ![original photo](https://github.com/andreea-ghe/Star-detection/blob/main/star4.png)
- ![first method data](https://github.com/andreea-ghe/Star-detection/blob/main/star1.png)
- ![first_method watershed](https://github.com/andreea-ghe/Star-detection/blob/main/star2.png)
- ![first method detected stars](https://github.com/andreea-ghe/Star-detection/blob/main/star3.png)
