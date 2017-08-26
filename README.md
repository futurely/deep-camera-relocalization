# Deep Camera Relocalization

## Getting Started

 * Download the Cambridge Landmarks King's College dataset from [here](https://www.repository.cam.ac.uk/handle/1810/251342).

 * Download the starting and trained weights from [here](https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md).

 * To run:
   * Extract the King's College dataset to wherever you prefer
   * Extract the starting and trained weights to wherever you prefer
   * If you want to retrain, simply run train.py
   * If you just want to test, simply run test.py 

## References

Ronald Clark, Sen Wang, Andrew Markham, Niki Trigoni, Hongkai Wen. VidLoc: A Deep Spatio-Temporal Model for 6-DoF Video-Clip Relocalization. CVPR 2017.

Alex Kendall and Roberto Cipolla. Geometric loss functions for camera pose regression with deep learning. CVPR, 2017.

Alex Kendall, Matthew Grimes and Roberto Cipolla. PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization. ICCV, 2015.

## Acknowledgement

Original implementation of PoseNet: https://github.com/kentsommer/tensorflow-posenet
