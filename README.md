Deepfake Detection Try-Out Model

This repository contains the try-out and improved version of my final year deepfake detection project. It was developed separately from the original version so that I could test alternative ideas, compare performance, and evaluate whether the changes improved real-world detection results.

Project Purpose

The purpose of this try-out version was to improve the performance of the original deepfake detector, especially when testing on videos outside the main training set. The original system performed well during training and validation, but some predictions on external videos were still incorrect. For that reason, I created this separate version to investigate whether changes in preprocessing and model testing would improve detection quality.

What This Version Includes

This version keeps the same overall idea of a Flask-based deepfake detection system, but it was used as an experimental version to test improvements. The system:

loads a trained convolutional neural network model
accepts uploaded video files through a Flask web interface
extracts frames from the uploaded video
preprocesses the frames before inference
predicts whether the input video is REAL or FAKE
stores testing results for evaluation and comparison
The try-out version was also connected to an Excel-based results table so that testing outputs could be saved and reviewed more easily.

Why This Version Was Created

The original version of the system achieved strong validation accuracy, but this did not always translate into the best performance on unseen videos. In particular, some real videos were classified as fake, while some fake videos were classified as real. To investigate this, I created this try-out version as a separate experimental branch of the project.

This allowed me to test new ideas without replacing the original baseline system. It also made it possible to directly compare both versions and decide which one performed better in practice.

Main Difference from the Original Version

The main difference is that this repository focuses on testing and improvement rather than only keeping the baseline implementation. It was used to experiment with an updated model setup and additional evaluation methods. While the original project remained important as the baseline reference, this try-out version was developed to see whether a more practical and reliable detector could be achieved.

Testing and Evaluation

To evaluate the try-out version, the model was tested on separate real and fake videos. The aim was not only to look at validation accuracy, but also to observe real inference performance on test samples outside the immediate training process. Testing results were recorded in structured tables for comparison with the original system.

This helped identify:

correct predictions
incorrect predictions
false positives
false negatives
overall practical behaviour of the model
The try-out version was useful because it provided a clearer picture of how the detector behaved beyond training accuracy alone.

Technologies Used

The project was developed using:

Python
Flask
PyTorch
OpenCV
HTML/CSS
OpenPyXL for test result recording
Repository Structure

This repository contains the code for the try-out version only. Large datasets, generated outputs, temporary uploads, and trained model weight files may be excluded from the repository through .gitignore to keep the project lightweight and suitable for GitHub submission.

Conclusion

This try-out repository represents the improved and experimental side of the deepfake detection project. It was created to test whether alternative implementation choices could produce better practical results than the original version. By separating this version from the original system, I was able to experiment more safely, compare results more clearly, and document the 
