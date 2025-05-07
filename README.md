# Cyber-Attack-Detection-Using-CNN
## Description

This project focuses on detecting cyber attacks in large-scale networks using machine learning techniques, specifically a Convolutional Neural Network (CNN) and other classifiers like Logistic Regression, Gradient Boosting, and Random Forest. The UNSW-NB15 dataset is used, which contains network traffic data with various attack types (e.g., DoS, Exploits, Fuzzers) and normal traffic. The project includes exploratory data analysis (EDA), data preprocessing (e.g., encoding, scaling, handling class imbalance with SMOTE), model training, and performance evaluation using metrics such as accuracy, precision, recall, and F1-score.

Uses the UNSW-NB15 dataset, which includes various attack types (DoS, Exploits, Fuzzers) and normal traffic.

Includes:

Exploratory Data Analysis (EDA)
Data preprocessing (encoding, scaling, SMOTE)
Model training and evaluation using accuracy, precision, recall, F1-score

## 🚀 Installation
Clone the repository
git clone https://github.com/your-username/cyber-attack-detection.git
cd cyber-attack-detection
(Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Requirements:

nginx
Copier
Modifier
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
tensorflow
keras-tuner
imblearn
pickle5

Download the dataset
Get the files from the UNSW-NB15 dataset on Kaggle.(https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
Place the following in the project root:

UNSW-NB15_1.csv to UNSW-NB15_4.csv

UNSW-NB15_LIST_EVENTS.csv

NUSW-NB15_features.csv

## 🧪 Usage
Run the Jupyter notebook
jupyter notebook
Open Notebook_CNN_to_detect_cyber_attacks_in_large_scale_networks.ipynb

Follow the steps in the notebook to:
Load and merge the dataset
Perform EDA
Encode and scale data
Handle class imbalance with SMOTE
Train and evaluate models (Logistic Regression, CNN, etc.)
Save the CNN model as model_CNN.pkl
Visualize metrics
Alternatively, convert to Python script and run it

jupyter nbconvert --to script Notebook_CNN_to_detect_cyber_attacks_in_large_scale_networks.ipynb
python Notebook_CNN_to_detect_cyber_attacks_in_large_scale_networks.py
## 🔍 Features
EDA
Visualizations using Seaborn and Matplotlib
Analysis of feature distribution and attack types
Preprocessing
Encoding: LabelEncoder, OneHotEncoder
Scaling: MinMaxScaler, StandardScaler
Imbalance handling: SMOTE, RandomUnderSampler
Optional: PCA for dimensionality reduction
Model Training
Logistic Regression on resampled data
CNN with Conv1D, tuned via Keras Tuner
Evaluation
Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
Confusion matrices
Model Persistence
Save trained CNN with pickle

## 📊 Dataset: UNSW-NB15
Files

UNSW-NB15_1.csv to UNSW-NB15_4.csv: Main traffic data (49 features)
UNSW-NB15_LIST_EVENTS.csv: Attack categories
NUSW-NB15_features.csv: Feature metadata
### Features
Numerical: packet/byte counts, durations
Categorical: protocol, service, state

### Labels
Binary: 0 (normal), 1 (attack)
Multi-class: Fuzzers, DoS, Exploits, etc.

## 🛠 Technologies
Language: Python 3.x
Libraries:
pandas, numpy: data handling
matplotlib, seaborn: visualization
scikit-learn: ML models, preprocessing
tensorflow, keras: deep learning
keras-tuner: hyperparameter tuning
imblearn: class imbalance tools
pickle: model serialization
