# Cyber-Attack-Detection-Using-CNN
Description

This project focuses on detecting cyber attacks in large-scale networks using machine learning techniques, specifically a Convolutional Neural Network (CNN) and other classifiers like Logistic Regression, Gradient Boosting, and Random Forest. The UNSW-NB15 dataset is used, which contains network traffic data with various attack types (e.g., DoS, Exploits, Fuzzers) and normal traffic. The project includes exploratory data analysis (EDA), data preprocessing (e.g., encoding, scaling, handling class imbalance with SMOTE), model training, and performance evaluation using metrics such as accuracy, precision, recall, and F1-score.

Installation

To set up the project locally, follow these steps:
Clone the repository:
git clone https://github.com/your-username/cyber-attack-detection.git
Navigate to the project directory:
cd cyber-attack-detection
Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

The requirements.txt file includes:

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



Download the dataset: Download the UNSW-NB15 dataset files (UNSW-NB15_1.csv, UNSW-NB15_2.csv, UNSW-NB15_3.csv, UNSW-NB15_4.csv, UNSW-NB15_LIST_EVENTS.csv, NUSW-NB15_features.csv) from the UNSW-NB15 dataset page(https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15). Place these files in the project root directory.

Usage
To run the project, execute the Jupyter Notebook:

Launch Jupyter Notebook:
jupyter notebook
Open the Notebook_CNN_to_detect_cyber_attacks_in_large_scale_networks.ipynb file in the browser.
Run the cells sequentially to:

Load and merge the UNSW-NB15 dataset.
Perform exploratory data analysis (EDA).
Preprocess the data (e.g., encoding categorical variables, scaling, handling class imbalance with SMOTE).
Train and evaluate models (Logistic Regression, CNN, etc.).
Save the trained CNN model as model_CNN.pkl.
Visualize performance metrics and results.
Alternatively, you can convert the notebook to a Python script and run it:
jupyter nbconvert --to script Notebook_CNN_to_detect_cyber_attacks_in_large_scale_networks.ipynb
python Notebook_CNN_to_detect_cyber_attacks_in_large_scale_networks.py

Features
Exploratory Data Analysis (EDA): Visualizations (e.g., using Seaborn, Matplotlib) and statistical analysis to understand the dataset's features and attack distributions.


Data Preprocessing:
Encoding categorical variables (e.g., using LabelEncoder, OneHotEncoder).
Scaling numerical features (e.g., using MinMaxScaler, StandardScaler).
Handling class imbalance with SMOTE and RandomUnderSampler.
Dimensionality reduction with PCA (if applicable).



Model Training:
Logistic Regression with resampled data for improved performance.
CNN with Conv1D layers, optimized using Keras Tuner for hyperparameter tuning.
Model Evaluation: Metrics including accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
Model Persistence: Saving the trained CNN model using pickle for future use.

Dataset
The UNSW-NB15 dataset is used, comprising:


Data Files:
UNSW-NB15_1.csv, UNSW-NB15_2.csv, UNSW-NB15_3.csv, UNSW-NB15_4.csv: Network traffic data with 49 features and labels (normal or attack).
UNSW-NB15_LIST_EVENTS.csv: Details of attack categories and subcategories.
NUSW-NB15_features.csv: Metadata describing the dataset's features.



Key Features:
Numerical features (e.g., packet counts, byte counts, durations).
Categorical features (e.g., protocol, service, state).
Labels: Binary (0 for normal, 1 for attack) or multi-class (specific attack types like Fuzzers, DoS, Exploits).
Attack Categories: Includes Fuzzers, Reconnaissance, Shellcode, Analysis, Backdoors, DoS, Exploits, and more.

Technologies Used
Python: 3.x
Libraries:
pandas, numpy: Data manipulation and numerical computations.
matplotlib, seaborn: Data visualization.
scipy: Statistical analysis.
scikit-learn: Machine learning models, preprocessing, and evaluation.
tensorflow, keras: CNN implementation and training.
keras-tuner: Hyperparameter optimization.
imblearn: Handling class imbalance (SMOTE, RandomUnderSampler).
pickle: Model serialization.
