# AI-Generated Text Detection Application

## Project Overview
This project aims to detect whether a given piece of text is generated by AI or written by a human. The model leverages various machine learning techniques including Support Vector Machines (SVM), Naive Bayes, Logistic Regression and Artificial Neural Networks (ANN) to achieve high accuracy. Also voting classifier is used to combine the predictions from Naive Bayes, Support Vector Machine and Logistic Regression to give prediction.

### Model Evaluation
![image](images.png)

- Naive Bayes Classifier

![img.png](img.png)

- Support Vector Machine

![img_1.png](img_1.png)

- Logistic Regression

![img_2.png](img_2.png)

- Voting Classifier

![img_3.png](img_3.png)

- ANN

![img_4.png](img_4.png)

### Database used
https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset

## Steps to run the program on Windows
1. Create a virtual environment 
```
python -m venv "your environment name here"
```
2. Activate the virtual environment
```
"your environment name here"\Scripts\activate.bat
```
3. Install all required libraries
```
pip install -r requirements.txt
```
4. Run the program
```
streamlit run app.py
```