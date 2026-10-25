QRS-Annotation & ECG Analysis Pipeline

A deep learning-based pipeline for electrocardiogram (ECG) signal processing, QRS complex detection, beat classification, and Atrial Fibrillation (AFib) run detection.

🚀 Overview & How it Works

This repository is designed to process continuous ECG signal data, detect heartbeats, and classify them using advanced deep learning architectures. The pipeline operates in three main stages:

Signal Processing & Extraction: Raw ECG records are processed to isolate the QRS complexes and extract continuous signal segments.

Hierarchical Classification: Instead of a flat multi-class approach, the system utilizes a Hierarchical QRS Classifier to first distinguish normal from abnormal beats, and then further classify specific arrhythmias.

AFib Detection & Annotation Overlap: Continuous segments are analyzed for Atrial Fibrillation runs. The system calculates the practical accuracy of the model by comparing the total time overlap of the model's detections directly against expert annotations (Total Detection Overlap / Total Annotation Time).

🧠 Model Architecture

The core of the classification engine is built using TensorFlow.

HierarchicalQRSClassifier: A layered neural network architecture that categorizes beats hierarchically, capturing both broad morphological differences and subtle beat-to-beat variations.

Threshold Optimization: The pipeline includes logic to evaluate prediction probability distributions and tune decision thresholds, ensuring the optimal balance between sensitivity and specificity for medical time-series data.

📊 Model Evaluation & Metrics

The model is evaluated by tracking exact predictions against ground-truth annotations (such as Record 03). Recent benchmark testing yielded the following performance metrics:

Accuracy: 86.65%

Specificity: 86.67%

Sensitivity: 83.37%

Confusion Matrix Breakdown:

True Positives (TP): 386

True Negatives (TN): 107,455

False Positives (FP): 16,533

False Negatives (FN): 77

(Note: These metrics can be reproduced or updated by running the built-in evaluation scripts.)

⚙️ Installation

Clone the repository:

git clone https://github.com/arianabdollahi83/QRS-Annotation.git
cd QRS-Annotation


Install the required dependencies (ensure you have TensorFlow, NumPy, and signal processing libraries like wfdb installed):

pip install -r requirements.txt


💻 Usage

1. Training and Evaluating the QRS Classifier

To train the hierarchical model, extract features, and evaluate its performance metrics, run the main training script. This script will output the updated confusion matrix and optimal thresholds.

python train_qrs_v3.py


2. Detecting AFib Runs

To analyze continuous signal data for Atrial Fibrillation runs and visualize the detections against the ground-truth annotations, use the detector script:

python afib_detector.py


This script handles the practical detection of AFib runs and plots the signal to visually verify the overlap between annotations and model predictions.
