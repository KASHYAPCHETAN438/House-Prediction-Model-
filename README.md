# 🏠 House Price Prediction System

A Machine Learning-based web application that predicts house prices based on various property features such as **area, bedrooms, bathrooms, floors, year built, location, condition, and garage availability**.

The project demonstrates the complete Machine Learning workflow from **data preprocessing and exploratory data analysis to model training, comparison, evaluation, model saving, and FastAPI deployment**.

---

## 📌 Project Overview

The objective of this project is to develop a regression-based Machine Learning system that can estimate the price of a house using its property-related features.

### Input Features

| Feature   | Description                       |
| --------- | --------------------------------- |
| Area      | Total area of the house           |
| Bedrooms  | Number of bedrooms                |
| Bathrooms | Number of bathrooms               |
| Floors    | Number of floors                  |
| YearBuilt | Year in which the house was built |
| Location  | Location/category of the property |
| Condition | Overall condition of the house    |
| Garage    | Garage availability               |

### 🎯 Target Variable

**Price** — Predicted selling price of the house.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
Model Comparison
   ↓
Model Evaluation
   ↓
Save Trained Models
   ↓
FastAPI Deployment
   ↓
Web Application
```

---

## 🤖 Machine Learning Models

Multiple regression algorithms are trained and compared to identify a suitable model for house price prediction.

The project can include models such as:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

The models are evaluated using regression metrics such as:

* **MAE** — Mean Absolute Error
* **MSE** — Mean Squared Error
* **RMSE** — Root Mean Squared Error
* **R² Score** — Coefficient of Determination

### 📊 Evaluation

The trained models are compared based on their prediction performance on the test dataset.

For a regression problem, a higher **R² Score** and lower **MAE/RMSE** generally indicate better predictive performance on the evaluation dataset.

---

## 📊 Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the dataset before model training.

The analysis includes:

* Dataset shape and structure
* Data types
* Missing-value analysis
* Duplicate-value detection
* Statistical summary
* Feature distributions
* Correlation analysis
* Outlier analysis
* Relationship between features and house prices

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

1. Load the dataset.
2. Inspect the dataset.
3. Handle missing values.
4. Remove duplicate records where required.
5. Separate input features and target variable.
6. Encode categorical features such as:

   * Location
   * Condition
   * Garage
7. Split the data into training and testing sets.
8. Apply the required feature transformation/scaling.
9. Prepare the processed data for model training.

---

## 🏗️ Project Structure

```text
house_price_app/
│
├── 📄 main.py
├── 📄 house_price_models.joblib
├── 📄 requirements.txt
├── 📄 README.md
│
├── 📁 templates/
│   └── index.html
│
└── 📁 static/
    └── style.css
```

### File Description

| File/Folder                 | Description                                         |
| --------------------------- | --------------------------------------------------- |
| `main.py`                   | FastAPI application and prediction logic            |
| `house_price_models.joblib` | Saved trained ML model(s) and preprocessing objects |
| `requirements.txt`          | Required Python dependencies                        |
| `templates/index.html`      | Web application interface                           |
| `static/style.css`          | Styling for the web application                     |
| `README.md`                 | Project documentation                               |

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Joblib

### Data Analysis

* Pandas
* NumPy
* Matplotlib
* Seaborn

### Backend

* FastAPI
* Uvicorn

### Frontend

* HTML5
* CSS3
* Jinja2 Templates

### Development Tools

* VS Code
* Jupyter Notebook / Google Colab
* Git
* GitHub

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/KASHYAPCHETAN438/House-Prediction-Model-.git
```

### 2. Navigate to the Project Directory

```bash
cd House-Prediction-Model-
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the FastAPI server using:

```bash
uvicorn main:app --reload
```

The application will start locally.

Open the application in your browser at:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

---

## 🔮 Prediction Process

The prediction process works as follows:

```text
User Input
    ↓
FastAPI
    ↓
Input Validation
    ↓
Data Preprocessing
    ↓
Trained ML Model
    ↓
Price Prediction
    ↓
Prediction Result
```

The user provides property information through the web interface.

The FastAPI backend receives the input, applies the same preprocessing used during model training, and passes the processed data to the trained Machine Learning model.

The predicted house price is then displayed to the user.

---

## 🌐 FastAPI API

FastAPI is used to create the backend API for the prediction system.

### API Documentation

After starting the application, visit:

```text
http://127.0.0.1:8000/docs
```

The Swagger UI allows you to:

* View available endpoints
* Enter prediction inputs
* Send API requests
* Test the prediction system
* View API responses

---

## 💾 Model Saving

After model training, the required trained model and preprocessing objects are saved using **Joblib**.

```text
house_price_models.joblib
```

This allows the application to load the already-trained model instead of training the model every time the FastAPI server starts.

---

## 📈 Model Evaluation

The project evaluates regression models using:

### MAE

Measures the average absolute difference between actual and predicted prices.

### MSE

Measures the average squared difference between actual and predicted prices.

### RMSE

Represents the square root of MSE and expresses prediction error in the same unit as the target variable.

### R² Score

Measures how much of the variation in house prices is explained by the model.

---

## ✨ Key Features

* 🏠 House price prediction
* 📊 Exploratory Data Analysis
* 🧹 Data cleaning and preprocessing
* 🔤 Categorical feature encoding
* 🤖 Multiple regression algorithms
* 📈 Model comparison
* 📏 Regression evaluation metrics
* 💾 Trained model serialization
* ⚡ FastAPI backend
* 🌐 Web-based prediction interface
* 📖 Interactive Swagger API documentation

---

## 🔐 Important Note

The prediction accuracy depends on the quality, size, distribution, and representativeness of the dataset used for training.

The predicted price should be considered a **Machine Learning estimate**, not a guaranteed market price.

---

## 🔮 Future Improvements

Possible improvements include:

* Add a larger and more diverse dataset
* Improve feature engineering
* Add additional property features
* Hyperparameter tuning
* Add model explainability
* Add database integration
* Create a REST API for external applications
* Deploy the application on cloud platforms
* Add authentication and user accounts
* Add prediction history
* Build an interactive dashboard

---

## 👨‍💻 Author

**Chetan Kashyap**

Computer Science Engineering Graduate

### Skills Demonstrated

* Python
* Machine Learning
* Data Analysis
* Regression
* FastAPI
* HTML & CSS
* Git & GitHub

## 🌐 Live Demo

[🚀 Visit Project Website](https://house-prediction-model-sfe6.onrender.com/)

---

## 📄 License

This project is created for **educational and portfolio purposes**.
