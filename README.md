# Oneshot-Face-Identify
This project is designed for one shot face identify people in video.

## Face Detection 
 - Face detection model used https://arxiv.org/pdf/1708.05234.pdf
 - Face identify [one shot learnig] model used https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf
 - Multi object tracking using SORT algorithm https://arxiv.org/pdf/1602.00763.pdf
 
 ## Run 
  - python App.py for [web view]
  - before run you should download the face compare weight file from the txt file in the folder
  - after you get valid all requirements you just run App.py
  - after you upload one video file through postman
  - check it will response true or not
  - http://0.0.0.0:5000/video_feed open this url in browser for output
  
 ## Requirement
  - numpy==1.17.2
  - tensorflow-gpu==1.14.0
  - opencv-python==4.1.2.30
  - scikit-learn==0.21.3
  - numba==0.45.1
  - filterpy==1.4.5
  - Flask==1.1.1
