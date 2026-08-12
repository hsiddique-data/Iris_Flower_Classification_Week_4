# 🌸 Iris Flower Classification Using Machine Learning

## Data Science Internship — Week 4

**Organization:** SystemTron  
**Project:** Iris Flower Classification  
**Project Type:** Supervised Machine Learning — Multi-Class Classification  
**Dataset:** Iris Flower Dataset  
**Primary Objective:** Classify Iris flowers into their respective species using sepal and petal measurements.

---

## 1. Executive Summary

The Iris Flower Classification project was completed as part of the Week 4 Data Science Internship task assigned by SystemTron.

The objective was to develop a supervised machine learning model capable of classifying Iris flowers into three species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

The classification was performed using four numerical measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The project followed a structured end-to-end Data Science workflow covering dataset acquisition, data understanding, data quality assessment, data cleaning, exploratory data analysis, feature preparation, train-test splitting, model development, model evaluation, model comparison, prediction, and model persistence.

Three machine learning algorithms were evaluated:

1. Decision Tree
2. Random Forest
3. K-Nearest Neighbors (KNN)

The models were evaluated using accuracy, precision, recall, F1-score, classification reports, and confusion matrices.

Among the three models, KNN achieved the highest performance on the held-out test dataset, achieving **100% accuracy, precision, recall, and F1-score** on the 30 test records.

The final KNN model was saved and successfully reloaded to verify that it could be reused for future predictions.

---

# 2. Business and Technical Objective

## 2.1 Problem Statement

Given measurements of an Iris flower, the objective is to automatically determine which species the flower belongs to.

The classification problem can be represented as:

**Input:**

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

**Output:**

- Iris-setosa
- Iris-versicolor
- Iris-virginica

This is a **multi-class supervised classification problem** because the model learns from labeled examples and predicts one of three possible classes.

---

## 2.2 Project Objectives

The main objectives of the project were:

- Acquire and verify the dataset provided for the internship task.
- Understand the structure and characteristics of the dataset.
- Identify missing values and duplicate records.
- Clean and prepare the dataset for machine learning.
- Perform exploratory data analysis.
- Identify useful relationships between the flower measurements.
- Train multiple classification algorithms.
- Evaluate and compare model performance.
- Select the best-performing model.
- Use the selected model to predict a new flower.
- Save the trained model for future reuse.
- Document the complete machine learning workflow.

---

# 3. Dataset

The Iris Flower Dataset provided for the SystemTron internship was obtained from the specified Kaggle dataset.

The dataset was downloaded and extracted into the project's raw data directory.

### Dataset Source

The dataset contains measurements of three Iris species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

### Dataset Location

```text
data/raw/IRIS.csv
```

The downloaded dataset was verified before beginning the analysis.

---

# 4. Project Structure

The project was organized using a structured and reproducible directory layout.

```text
Iris_Flower_Classification_Week_4/
│
├── data/
│   ├── raw/
│   │   ├── iris-flower-dataset.zip
│   │   └── IRIS.csv
│   │
│   └── processed/
│       └── iris_dataset_cleaned.csv
│
├── notebooks/
│   └── Iris_Flower_Classification.ipynb
│
├── reports/
│   └── Iris_Flower_Classification_Report.md
│
├── results/
│   ├── visualization files
│   └── confusion matrices
│
├── models/
│   ├── iris_knn_model.pkl
│   ├── model_comparison.csv
│   └── new_flower_prediction.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

This separation keeps raw data, processed data, notebooks, reports, visualizations, and trained models organized independently.

---

# 5. Data Understanding

The dataset was loaded using Pandas and examined before performing any transformation.

## 5.1 Dataset Dimensions

The original dataset contained:

| Property | Value |
|---|---:|
| Rows | 150 |
| Columns | 5 |
| Numerical Features | 4 |
| Target Variable | 1 |

The five columns were:

```text
sepal_length
sepal_width
petal_length
petal_width
species
```

---

## 5.2 Data Types

The four measurement columns were stored as numerical floating-point values.

The species column was categorical.

| Column | Data Type |
|---|---|
| sepal_length | float64 |
| sepal_width | float64 |
| petal_length | float64 |
| petal_width | float64 |
| species | string |

---

## 5.3 Statistical Summary

The descriptive statistics showed the following ranges:

| Feature | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| Sepal Length | 5.84 | 4.3 | 7.9 |
| Sepal Width | 3.05 | 2.0 | 4.4 |
| Petal Length | 3.76 | 1.0 | 6.9 |
| Petal Width | 1.20 | 0.1 | 2.5 |

The statistics provided an initial understanding of the scale and distribution of the numerical variables.

---

# 6. Data Quality Assessment

Before model development, the dataset was checked for missing values and duplicate records.

## 6.1 Missing Values

The missing-value analysis showed:

```text
Total Missing Values: 0
```

Therefore, no imputation or missing-value treatment was required.

The dataset was ready for the next cleaning stage.

---

## 6.2 Duplicate Records

Three duplicate records were identified.

```text
Duplicate records before cleaning: 3
```

The duplicate observations were removed to prevent repeated records from influencing the model.

After cleaning:

```text
Duplicate records after cleaning: 0
```

The final cleaned dataset contained:

```text
147 records
5 columns
```

---

# 7. Data Preprocessing

After data quality assessment, the dataset was prepared for machine learning.

## 7.1 Feature Selection

The four measurement columns were selected as independent variables:

```text
sepal_length
sepal_width
petal_length
petal_width
```

The target variable was:

```text
species
```

---

## 7.2 Target Encoding

Machine learning algorithms require numerical representations of categorical target values.

The species labels were encoded as follows:

| Species | Encoded Value |
|---|---:|
| Iris-setosa | 0 |
| Iris-versicolor | 1 |
| Iris-virginica | 2 |

The encoded target column was named:

```text
species_encoded
```

---

## 7.3 Final Feature Matrix

After preprocessing:

```text
Feature Shape: (147, 4)
Target Shape: (147,)
```

The processed dataset was saved to:

```text
data/processed/iris_dataset_cleaned.csv
```

---

# 8. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the distribution of the Iris species and the relationships between the numerical features.

The analysis included:

1. Species distribution
2. Feature distributions
3. Feature comparison using boxplots
4. Feature relationship analysis
5. Correlation analysis

All important visualizations were saved in the `results/` directory.

---

# 9. Species Distribution

The species distribution was visualized to understand the number of observations belonging to each class.

![Species Distribution](../results/01_species_distribution.png)

### Interpretation

The cleaned dataset contains observations from all three Iris species.

The distribution analysis confirms that all three classes are represented in the dataset and are suitable for multi-class classification.

---

# 10. Feature Distributions

The distributions of the numerical measurements were examined to understand their ranges and variation.

![Feature Distributions](../results/02_feature_distribution.png)

### Interpretation

The feature distributions show noticeable differences in the measurement ranges.

In particular, petal-related measurements show greater variation between Iris species than several sepal measurements.

These differences are useful for classification because machine learning algorithms can use them to distinguish between species.

---

# 11. Feature Comparison

Boxplots were used to examine the distribution of numerical measurements across the Iris species.

![Feature Boxplots](../results/03_petal_length_boxplot.png)

### Interpretation

The boxplots demonstrate that petal measurements provide strong separation between the Iris species.

Iris-setosa has considerably smaller petal measurements compared with Iris-versicolor and Iris-virginica.

The overlap between Versicolor and Virginica is greater, which explains why these two classes are more challenging to distinguish.

---

# 12. Feature Relationship Analysis

The relationship between petal measurements was examined using scatter plots.

![Feature Relationship](../results/04_petal_length_vs_width.png)

### Interpretation

Petal length and petal width show a strong relationship.

Iris-setosa forms a clearly separated group, while Iris-versicolor and Iris-virginica are closer to each other.

This visual separation provides evidence that the selected features contain useful information for classification.

---

# 13. Correlation Analysis

A correlation analysis was performed to understand relationships between the numerical features.

![Correlation Heatmap](../results/05_correlation_heatmap.png)

### Interpretation

The correlation analysis indicates strong relationships among several flower measurements.

The petal measurements show particularly strong correlation with each other.

This confirms that the selected numerical features contain meaningful patterns that can be exploited by classification algorithms.

---

# 14. Train-Test Split

Before model training, the cleaned dataset was divided into training and testing datasets.

An **80:20 train-test split** was used.

| Dataset | Records |
|---|---:|
| Training Set | 117 |
| Testing Set | 30 |
| Total | 147 |

Stratified splitting was used to maintain a similar representation of the three Iris species in both datasets.

The test dataset was kept separate from model training and was used only for evaluating the final performance.

---

# 15. Machine Learning Methodology

Three supervised classification algorithms were selected.

## 15.1 Decision Tree

The Decision Tree algorithm learns a sequence of decision rules based on feature values.

It provides an interpretable classification approach and can model non-linear relationships.

---

## 15.2 Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

Each tree contributes to the final prediction, helping improve generalization compared with a single decision tree.

---

## 15.3 K-Nearest Neighbors

K-Nearest Neighbors classifies an observation based on the classes of its nearest neighboring observations.

A value of:

```text
K = 5
```

was used for the KNN model.

---

# 16. Model Training

All three models were trained using the 117 training observations.

The trained models were then used to generate predictions for the 30 unseen test observations.

All three models successfully generated predictions.

---

# 17. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

These metrics provide both overall and class-level information about model performance.

---

# 18. Model Performance Comparison

The evaluation results were:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Decision Tree | 93.33% | 93.33% | 93.33% | 93.33% |
| Random Forest | 96.67% | 96.97% | 96.67% | 96.66% |
| **KNN** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

---

# 19. Decision Tree Evaluation

The Decision Tree achieved:

- Accuracy: **93.33%**
- Precision: **93.33%**
- Recall: **93.33%**
- F1-score: **93.33%**

![Decision Tree Confusion Matrix](../results/06_decision_tree_confusion_matrix.png)

### Interpretation

The Decision Tree correctly classified all Iris-setosa observations in the test set.

Some confusion occurred between Iris-versicolor and Iris-virginica.

This indicates that these two species have more overlapping feature characteristics compared with Iris-setosa.

---

# 20. Random Forest Evaluation

The Random Forest achieved:

- Accuracy: **96.67%**
- Precision: **96.97%**
- Recall: **96.67%**
- F1-score: **96.66%**

![Random Forest Confusion Matrix](../results/07_random_forest_confusion_matrix.png)

### Interpretation

Random Forest improved upon the Decision Tree.

The model correctly classified all Iris-setosa observations and achieved strong performance on both Iris-versicolor and Iris-virginica.

Only a small amount of confusion remained between the latter two classes.

---

# 21. KNN Evaluation

The KNN model achieved:

- Accuracy: **100.00%**
- Precision: **100.00%**
- Recall: **100.00%**
- F1-score: **100.00%**

![KNN Confusion Matrix](../results/08_knn_confusion_matrix.png)

### Interpretation

KNN correctly classified all 30 observations in the held-out test dataset.

The confusion matrix contains correct predictions for all three species with no misclassified observations.

Therefore, KNN achieved the highest performance among the models evaluated in this project.

---

# 22. Best Model Selection

Based on the evaluation results, **K-Nearest Neighbors (KNN)** was selected as the final model.

### Final Model Performance

```text
Accuracy  : 100%
Precision : 100%
Recall    : 100%
F1-Score  : 100%
```

It is important to note that these results represent performance on the **30-record held-out test dataset** used in this project.

The result should therefore not be interpreted as a guarantee of 100% accuracy on all future Iris flowers.

---

# 23. New Flower Prediction

The selected KNN model was tested with a new flower sample.

### Input Measurements

| Measurement | Value |
|---|---:|
| Sepal Length | 5.9 |
| Sepal Width | 3.0 |
| Petal Length | 5.1 |
| Petal Width | 1.8 |

The model predicted:

## **Iris-virginica**

The prediction was generated using the trained KNN model.

The result was saved to:

```text
models/new_flower_prediction.csv
```

---

# 24. Model Persistence

The final KNN model was saved using Joblib.

```text
models/iris_knn_model.pkl
```

The saved model was subsequently loaded back into Python and used to generate a prediction.

The prediction from the loaded model matched the original model prediction.

This confirms that the trained model can be persisted and reused without retraining it.

---

# 25. Project Artifacts

## Dataset

```text
data/raw/IRIS.csv
data/processed/iris_dataset_cleaned.csv
```

## Notebook

```text
notebooks/Iris_Flower_Classification.ipynb
```

## Visualizations

```text
results/
├── 01_species_distribution.png
├── 02_feature_distribution.png
├── 03_petal_length_boxplot.png
├── 04_petal_length_vs_width.png
├── 05_correlation_heatmap.png
├── 06_decision_tree_confusion_matrix.png
├── 07_random_forest_confusion_matrix.png
└── 08_knn_confusion_matrix.png
```

## Model Outputs

```text
models/
├── iris_knn_model.pkl
├── model_comparison.csv
└── new_flower_prediction.csv
```

---

# 26. Technology Stack

The project was implemented using Python and the following libraries:

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| Joblib | Model persistence |
| Jupyter Notebook | Development and documentation |

---

# 27. Key Findings

The project produced several important findings:

1. The dataset contained 150 original observations.
2. No missing values were present.
3. Three duplicate records were identified and removed.
4. The final cleaned dataset contained 147 observations.
5. Petal measurements provided strong discriminatory information between species.
6. Iris-setosa was comparatively easy to distinguish from the other species.
7. Iris-versicolor and Iris-virginica showed greater overlap.
8. Random Forest performed better than the individual Decision Tree.
9. KNN achieved the highest test-set performance.
10. The trained KNN model successfully classified a new flower as Iris-virginica.
11. The trained model was successfully saved and reloaded for future predictions.

---

# 28. Conclusion

The Iris Flower Classification project successfully implemented an end-to-end supervised machine learning workflow.

The project began with acquisition and verification of the SystemTron-provided Iris dataset and continued through data understanding, quality assessment, cleaning, feature preparation, exploratory analysis, model training, evaluation, model selection, prediction, and model persistence.

Three classification algorithms were evaluated:

- Decision Tree
- Random Forest
- K-Nearest Neighbors

Among these models, KNN achieved the strongest performance on the held-out test dataset, with 100% accuracy, precision, recall, and F1-score.

The project also demonstrated practical model deployment preparation by saving the trained KNN model and successfully loading it again for prediction.

Overall, the project fulfills the SystemTron Week 4 internship objective of developing a machine learning model to classify Iris flowers based on their sepal and petal measurements.

---

# 29. Future Improvements

Although the current model performed strongly on the test dataset, the project can be extended through:

- KNN hyperparameter optimization
- Cross-validation
- GridSearchCV or RandomizedSearchCV
- Testing Logistic Regression
- Testing Support Vector Machines
- Testing Gradient Boosting models
- Feature scaling experiments for KNN
- Larger external validation datasets
- Model explainability techniques
- Interactive prediction dashboards
- Deployment using Flask, FastAPI, or Streamlit
- Containerization and cloud deployment

These improvements would help evaluate model robustness and prepare the solution for a more production-oriented environment.

---

# 30. Final Project Status

| Component | Status |
|---|---|
| Dataset Acquisition | ✅ Completed |
| Data Understanding | ✅ Completed |
| Data Quality Assessment | ✅ Completed |
| Data Cleaning | ✅ Completed |
| Feature Preparation | ✅ Completed |
| Exploratory Data Analysis | ✅ Completed |
| Train-Test Split | ✅ Completed |
| Model Training | ✅ Completed |
| Model Evaluation | ✅ Completed |
| Model Comparison | ✅ Completed |
| New Flower Prediction | ✅ Completed |
| Model Persistence | ✅ Completed |
| Project Report | ✅ Completed |

---

## Final Result

**Selected Model:** K-Nearest Neighbors (KNN)

**Test Accuracy:** **100%**

**Prediction Example:** **Iris-virginica**

**Model File:** `models/iris_knn_model.pkl`

**Project Status:** **Completed ✅**
