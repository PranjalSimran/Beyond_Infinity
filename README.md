### AI POWERED HEALTH ISSUE PREDICTION & PERSONALIZED PREVENTION SYSTEM

# Project Overview:
The project focuses on developing a machine learning-based system to predict diseases like Monkeypox, Skin Cancer, Heart Disease, Diabetes, and HIV/AIDS based on user data (e.g., symptoms, medical history, and lifestyle). Along with prediction, the system will provide relevant preventions and precautions for each disease to promote early intervention and better health management.

# Key Features:

1. Disease Prediction: Accurate identification of potential diseases using machine learning models.
2. Prevention and Precaution Guidance: Personalized health advice to prevent or manage the predicted diseases.

# Heart-Disease-Prediction-using-Machine-Learning
The project involved analysis of the heart disease patient dataset with proper data processing. Then, different models were trained and and predictions are made with different algorithms KNN, Decision Tree, Random Forest,SVM,Logistic Regression etc
This is the jupyter notebook code and dataset I've used for my Kaggle kernel 'Binary Classification with Sklearn and Keras'

I've used a variety of Machine Learning algorithms, implemented in Python, to predict the presence of heart disease in a patient. This is a classification problem, with input features as a variety of parameters, and the target variable as a binary variable, predicting whether heart disease is present or not.

Machine Learning algorithms used:

1. Logistic Regression (Scikit-learn)
2. Naive Bayes (Scikit-learn)
3. Support Vector Machine (Linear) (Scikit-learn)
4. K-Nearest Neighbours (Scikit-learn)
5. Decision Tree (Scikit-learn)
6. Random Forest (Scikit-learn)
7. XGBoost (Scikit-learn)
8. Artificial Neural Network with 1 Hidden layer (Keras)

Accuracy achieved: 95% (Random Forest)

Dataset used: https://www.kaggle.com/ronitf/heart-disease-uci

# Diabetes Prediction

The "Diabetes Prediction" project aims to develop a model that can predict whether an individual is likely to have diabetes based on various features. This prediction task holds significant importance in healthcare, as early detection and intervention can lead to better management and treatment outcomes. By employing machine learning algorithms and a carefully curated dataset, this project provides an effective means of predicting diabetes.

## Key Features

- **Data Collection and Processing:** The project involves collecting a dataset containing features related to individuals' health, such as glucose levels, blood pressure, BMI, and more. Using Pandas, the collected data is cleaned, preprocessed, and transformed to ensure it is suitable for analysis. The dataset is included in the repository for easy access.

- **Data Visualization:** The project utilizes data visualization techniques to gain insights into the dataset. By employing Matplotlib or Seaborn, visualizations such as histograms, box plots, and correlation matrices are created. These visualizations provide a deeper understanding of the data distribution and relationships between features.

- **Train-Test Split:** To evaluate the performance of the classification model, the project employs the train-test split technique. The dataset is divided into training and testing subsets, ensuring that the model is trained on a portion of the data and evaluated on unseen data. This allows for an accurate assessment of the model's ability to generalize to new data.

- **Feature Scaling:** As part of the preprocessing pipeline, the project utilizes the StandardScaler from Scikit-learn to standardize the feature values. Standardization ensures that all features have a mean of 0 and a standard deviation of 1, which can help improve the performance and convergence of certain machine learning algorithms.

- **Support Vector Machine Model:** The project employs Support Vector Machines (SVM), a powerful supervised learning algorithm, to build the classification model. SVM is known for its ability to handle high-dimensional data and nonlinear relationships. The Scikit-learn library provides an implementation of SVM that is utilized in this project.

- **Model Evaluation:** To assess the performance of the SVM model, the project employs various evaluation metrics such as accuracy, precision, recall, and F1-score. These metrics provide insights into the model's ability to correctly classify individuals with and without diabetes.

# SkinCancer-Machine_Learning_Project

- Dataset: Skin Cancer MNIST: HAM10000 (https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000?select=hmnist_28_28_L.csv)

![data](https://challenge2018.isic-archive.com/wp-content/uploads/2018/04/task3.png)

## Techniques
1. PCA 
2. Data balancing
3. Resampling (Oversampling and Undersampling)
4. Random Undersampling
5. Random Oversampling
# Challenge results
Logistic Regression | 50.23%
XGBoosting| 48.85%
SVM|14.28%

### AIDS:
Predicting AIDS Virus Infection Using Machine Learning: Achieving 95% Accuracy with Logistic Regression and Random Forest Models

### Description:
This project aims to leverage advanced machine learning techniques to predict AIDS virus infection with high accuracy. Utilizing logistic regression and random forest algorithms, the study strives to achieve a prediction accuracy of 95%. The project involves data preprocessing, feature selection, model training, and evaluation to create robust predictive models. By comparing the performance of logistic regression and random forest, this project seeks to identify the most effective approach for predicting AIDS virus infection, ultimately contributing to improved early detection and treatment strategies.

### Objective:
- To preprocess and prepare a dataset for predicting AIDS virus infection.
- To train logistic regression and random forest models using the prepared dataset.
- To evaluate the performance of both models and achieve a prediction accuracy of at least 95%.
- To compare the effectiveness of logistic regression and random forest in predicting AIDS virus infection.
- To provide insights and recommendations based on the model comparison for practical application in early detection and intervention strategies.

# Monkey-Pox-Prediction
Monkeypox, also known as mpox, is a rare infectious disease caused by the mpox virus. Predominant in Africa, this viral zoonotic disease  include symptoms like fever, headache, muscle aches and backache, swollen lymph nodes, chills and exhaustion. Early and accurate diagnosis is crucial for containment and patient management. With this repository I am joining people in presenting a machine learning model for detecting the monkeypox disease using simple algorithms like KNN, Random Forest and Decision Tree. This model achieves an aggregate accuracy of 76% only. I tried neither fine tuning/ hyperparameter nor deep learning algorithms to enhance the performance of the predictive model. I will update repo once I achieve a higher score of accuracy.

# Data Source
https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset/download?datasetVersionNumber=4

# Libraries used
pandas, numpy, matplotlib, seaborn, scikitlearn

# ALgorithms used
KNN, Random Forest and Decision Tree

# Data Preprocessing, Splitting and Model Training
*	Handle missing values
*	Inspect sample column entries
*	Balance target labels if any class imbalances
*	Split training data and test data in the ratio 8:2.
*	Train the training data using the mentioned three ML algorithms

# Model Performance 
*	Evaluated the accuracy, FI score, ROC curve and confusion matrix for each model
*	Got an aggregate score of 0.76
## Accuracy
Accuracy score of 76%.

## Acknowledgements

This project is made possible by the contributions of the open-source community and the powerful libraries it provides, including NumPy, Pandas, Scikit-learn, and Matplotlib. I extend my gratitude to the developers and maintainers of these libraries for their valuable work. In addition, the mentor Siddhardan, visit his channel here : https://www.youtube.com/@Siddhardhan

## Authors
- Samir Salman
- Simone Giorgioni



