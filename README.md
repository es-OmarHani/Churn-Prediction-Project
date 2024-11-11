# Customer Churn Prediction Project

## Project Overview
The **Customer Churn Prediction Project** aims to analyze customer behavior and predict churn using various machine learning models. By leveraging data visualization, feature engineering, and model tuning techniques, this project helps businesses identify customers likely to churn, enabling them to take proactive measures to improve retention rates. The project also includes a web application for real-time predictions, deployed using FastAPI.

## Key Objectives
- Analyze customer churn data to uncover patterns and insights.
- Perform data visualization using univariate and bivariate analyses.
- Build machine learning pipelines for preprocessing both numerical and categorical data.
- Evaluate multiple machine learning models for predicting customer churn.
- Fine-tune models for optimal performance.
- Deploy the best-performing model using FastAPI for real-time predictions.

## Dataset
The dataset used in this project contains information on customer demographics, account details, and service usage. Key features include:
- **CustomerID**: Unique identifier for each customer.
- **Demographic Information**: Age, gender, location, etc.
- **Account Details**: Contract type, payment method, tenure, etc.
- **Service Usage**: Internet, phone, streaming services, etc.
- **Churn**: Binary indicator (Yes/No) if a customer has churned.

## Data Analysis & Visualization
Data analysis was conducted to understand the distribution of features and their relationships with churn. Key steps included:
- **Univariate Analysis**: Examining individual features to understand their distributions.
- **Bivariate Analysis**: Analyzing the relationships between features and churn status.
- **Data Visualizations**: Created visualizations using tools like Matplotlib and Seaborn to highlight key insights.

## Feature Engineering & Preprocessing
To prepare the data for machine learning, various preprocessing techniques were applied:
- **Handling Missing Values**: Imputed missing data where necessary.
- **Scaling Numerical Features**: Used pipelines to apply scalers like `StandardScaler` and `MinMaxScaler`.
- **Encoding Categorical Features**: Utilized techniques such as `OneHotEncoder` and `LabelEncoder`.
- **Pipeline Integration**: Combined scaling and encoding into a unified pipeline for streamlined preprocessing.

## Model Selection & Evaluation
Several machine learning models were trained and evaluated for predicting churn:
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Random Forest Classifier**

### Model Tuning & Optimization
- **Hyperparameter Tuning**: Applied Grid Search and Random Search to find the best parameters.
- **Cross-Validation**: Ensured model robustness by using techniques like K-Fold cross-validation.
- **Model Evaluation**: Assessed models using metrics such as accuracy, precision, recall, and F1-score.

## Model Deployment with FastAPI
To make the churn prediction model accessible, a RESTful API was built using **FastAPI**:
- **Endpoints**: Created endpoints for prediction and health checks.
- **Deployment**: Hosted the FastAPI application on a local server for demonstration.
- **Input Handling**: Accepts customer data in JSON format and returns churn predictions.


## Key Results
- Achieved **high accuracy and F1-scores** with the Random Forest model after hyperparameter tuning.
- The FastAPI deployment allows for **quick and scalable predictions** for real-time use cases.

## Future Work
- **Feature Selection**: Apply techniques like PCA to reduce dimensionality.
- **Web Application**: Enhance the FastAPI app with a frontend using frameworks like Streamlit or Gradio.
- **Deployment on Cloud**: Deploy the model on cloud platforms like AWS or Azure for scalability.

## Tools & Technologies
- **Python**: Programming language used for the entire project.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Matplotlib & Seaborn**: Data visualization libraries.
- **Scikit-Learn**: Machine learning and model evaluation.
- **FastAPI**: Framework for building the web API.
- **Jupyter Notebooks**: For data analysis and model development.

## How to Run the Project
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd churn-prediction

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt

3. **Run the FastAPI Application:**

    ```bash
    uvicorn app.main:app --reload

3. **Access the API:**

    Open your browser and go to http://127.0.0.1:8000/docs for API documentation.