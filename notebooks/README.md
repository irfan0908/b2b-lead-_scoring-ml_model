B2B Lead Scoring Model 📈
This project is a machine learning model designed to help sales teams prioritize high-quality leads. It predicts the likelihood of a lead converting into a customer, allowing businesses to focus their efforts on the most promising prospects. The solution is built using Python and popular data science libraries like Scikit-learn, Pandas, and Matplotlib.

🚀 Key Features
Automated Data Ingestion: The script automatically downloads the required dataset from Kaggle using the Kaggle API.

Comprehensive Data Preprocessing: It handles missing values, encodes categorical features, and scales numerical data to prepare it for model training.

Dual-Model Approach: The pipeline trains and evaluates two distinct classification models:

Logistic Regression: A simple yet effective baseline model.

Random Forest Classifier: A powerful ensemble model known for its high accuracy.

Detailed Evaluation: The script provides a comprehensive evaluation of each model, including accuracy score, a classification report, and a confusion matrix heatmap for clear visualization of performance.

⚙️ How to Run the Code
Prerequisites
Ensure you have Python 3.x and the following libraries installed:

pandas

scikit-learn

matplotlib

seaborn

kaggle

You can install these using pip:

Bash

pip install pandas scikit-learn matplotlib seaborn kaggle
Kaggle API Setup
This project uses the Kaggle API to download the dataset. You must set up your API credentials to run the code.

Go to your Kaggle account settings and click on "Create New API Token." This will download a file named kaggle.json.

Move this kaggle.json file to the .kaggle directory in your home folder.

On Windows: C:\Users\<username>\.kaggle\

On macOS/Linux: ~/.kaggle/

Execution
Once the dependencies and Kaggle API are set up, run the script from your terminal:

Bash

python your_script_name.py
(Note: Replace your_script_name.py with the actual name of your Python file, for example, lead_scoring_model.py)

The script will download the data, preprocess it, train both models, and display the evaluation results and confusion matrix plots.

📊 Results
The script will print the performance metrics for both Logistic Regression and Random Forest models. The confusion matrix plots will also be generated, showing how each model performed in classifying leads.

📂 Project Structure
.
├── lead_scoring_model.py
├── data/
├── .gitignore
└── README.md



## Model Evaluation Results

After running the script, the following evaluation results were generated for both the Logistic Regression and Random Forest models.

### Logistic Regression
```text
🔹 Logistic Regression Results:
Accuracy: 0.8495670995670996
              precision    recall  f1-score   support

           0       0.85      0.91      0.88      1107
           1       0.85      0.76      0.80       741

    accuracy                           0.85      1848
   macro avg       0.85      0.83      0.84      1848
weighted avg       0.85      0.85      0.85      1848


🔹 Random Forest Results:
Accuracy: 0.9345238095238095
              precision    recall  f1-score   support

           0       0.93      0.97      0.95      1107
           1       0.95      0.89      0.92       741

    accuracy                           0.93      1848
   macro avg       0.94      0.93      0.93      1848
weighted avg       0.93      0.93      0.93      1848

Conclusion
Based on the results, the Random Forest Classifier demonstrates superior performance with an accuracy of 93.45% compared to the Logistic Regression model's 84.96%. The higher precision, recall, and f1-scores for both classes (0 and 1) indicate that the Random Forest model is more effective at correctly identifying and classifying leads.