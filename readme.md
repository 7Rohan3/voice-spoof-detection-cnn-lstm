# Voice Spoof Detection using CNN-LSTM

## Project Description

Voice spoofing attacks pose a significant threat to modern voice authentication systems used in banking, smart assistants, and biometric security applications. This project presents a deep learning-based voice spoof detection system that classifies speech signals as genuine or spoofed using Mel Frequency Cepstral Coefficients (MFCC) and a hybrid Convolutional Neural Network–Long Short-Term Memory (CNN–LSTM) architecture.

The system extracts MFCC features from audio signals and processes them using convolutional layers for spatial feature learning and LSTM layers for temporal sequence modeling. The proposed model demonstrates strong classification performance on a balanced subset of the ASVspoof 2017 dataset.

This project highlights the effectiveness of combining signal processing techniques with deep learning models to detect spoofed speech and improve voice authentication reliability.

---

## Objectives

* To develop a voice spoof detection system using deep learning techniques
* To extract MFCC features from speech signals
* To implement a CNN–LSTM hybrid model
* To evaluate model performance using confusion matrix and classification metrics
* To analyze accuracy and loss trends during training

---

## Dataset Used

* Dataset Name: ASVspoof 2017
* Total Samples Used: 1000

  * Genuine Speech: 500
  * Spoofed Speech: 500
* Data Split:

  * Training: 80%
  * Testing: 20%

A subset of the dataset was selected due to hardware and memory constraints while maintaining class balance.

---

## Model Architecture

The proposed model consists of the following stages:

1. Audio Input
2. MFCC Feature Extraction
3. Convolutional Neural Network (CNN)
4. Max Pooling Layer
5. Long Short-Term Memory (LSTM) Layer
6. Dense Output Layer

This hybrid architecture enables both spatial and temporal feature learning.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* Librosa
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## Results

The trained model achieved strong classification performance:

* Accuracy: 98.7%
* Precision: 98.5%
* Recall: 98.9%
* F1-Score: 98.7%
* Equal Error Rate (EER): ~1.9%

The following evaluation outputs were generated:

* Accuracy Curve
* Loss Curve
* Confusion Matrix

---

## Project Structure

Voice-Spoof-Detection
│
├── src/
│   ├── dataset_prepare.py
│   ├── feature_extract.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│
├── results/
├── requirements.txt
├── README.md
├── .gitignore

---

## How to Run the Project

Install dependencies:

pip install -r requirements.txt

Run project pipeline:

python src/dataset_prepare.py
python src/feature_extract.py
python src/train.py
python src/evaluate.py

---

## Applications

* Voice authentication systems
* Banking security
* Smart assistants
* Biometric identification
* Anti-spoofing systems

---

## Author

Rohan
Akash 
Ishan
CSE (AI & ML)
Chandigarh University
