# Explainable AI for Autonomous Vehicle Safety

**Learning from California DMV Disengagement and Collision Reports**  
**UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence — Capstone Project**

## Project Overview

Autonomous vehicle safety is an important consideration as self-driving technology develops. This project uses publicly available California Department of Motor Vehicles (DMV) records to explore patterns in reported autonomous vehicle disengagements and collisions.

The central task is to **classify historical records as disengagement reports or collision reports**, rather than predict future crashes. I compare multiple machine learning methods and examine which available features contribute to their predictions.

## Research Question

**Can machine learning models identify patterns in historical autonomous vehicle disengagement and collision reports and distinguish between these two types of reported events?**

## Business Understanding

Collision reports are much less frequent than disengagement reports in the modeling dataset. A model can therefore achieve high overall accuracy by predicting the majority class while missing the collision reports. I focus on **recall**, **precision**, and **F1-score** to assess whether models can identify the minority class without generating excessive false alarms.

The analysis can help explain the strengths and limitations of using historical public reporting data for safety-related research. It does **not** provide a validated manufacturer safety ranking or an estimate of future collision probability.

## Data Sources

The project uses California DMV autonomous vehicle datasets:

- **Disengagement reports:** Reported events in which autonomous driving was disengaged.
- **Collision reports:** Reported autonomous vehicle collision events.
- **Autonomous vehicle mileage reports:** Context for understanding differences in testing activity across manufacturers.

The classification analysis focuses on **2024 reports**, with **1,457 disengagement records and 101 collision records** (1,558 total). The held-out test set contains **312 records, including 20 collision reports**.

**Source:** [California DMV Autonomous Vehicles](https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/).

## Methodology (CRISP-DM)

### 1. Business and Data Understanding

I reviewed the available DMV records and explored patterns by manufacturer, time, mileage, collision type, and vehicle operating mode. I also examined missing data, outliers, and reporting inconsistencies.

### 2. Data Preparation

I cleaned and standardized manufacturer names and date fields, addressed relevant data-quality issues, and prepared a common set of predictors available for both report types. The classification models use **manufacturer** and **reporting month**. The target indicates whether a record is a collision or disengagement report.

### 3. Exploratory Data Analysis

The notebook includes visualizations of disengagements, testing mileage, disengagement rates, collision counts, collision types, operating modes, and reporting patterns over time. Raw event counts are not interpreted as direct measures of manufacturer safety.

### 4. Modeling

I evaluated multiple classification approaches:

- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Gradient Boosting
- Neural Network (MLPClassifier), including a class-balanced version

I also used **5-fold stratified cross-validation**, **GridSearchCV**, and **RandomizedSearchCV** for model evaluation and hyperparameter tuning.

### 5. Evaluation

The principal evaluation metrics are:

- **Recall:** The share of collision reports correctly identified; important because missing collision reports is undesirable.
- **Precision:** The share of predicted collision reports that actually are collision reports; important for understanding false positives.
- **F1-score:** A combined measure of precision and recall, useful for comparing models on imbalanced data.
- **Accuracy:** Reported for completeness, but not used alone to select a model.

## Model Results

### Initial model comparison

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| RandomizedSearchCV Random Forest | 0.2742 | 0.85 | **0.4146** |
| Random Forest | 0.2712 | 0.80 | 0.4051 |
| GridSearchCV Random Forest | 0.2667 | 0.80 | 0.4000 |
| Gradient Boosting | 0.2368 | 0.90 | 0.3750 |
| Decision Tree | 0.2222 | 0.80 | 0.3478 |
| Balanced Neural Network | 0.2118 | 0.90 | 0.3429 |
| Logistic Regression | 0.1977 | 0.85 | 0.3208 |
| Original Neural Network | 0.0000 | 0.00 | 0.0000 |

The RandomizedSearchCV Random Forest had the highest **reported test F1-score (0.4146)** in the initial comparison. The Gradient Boosting and balanced Neural Network models each identified 90% of collision reports but generated more false positives.

The original Neural Network achieved **93.59% accuracy** while identifying **none** of the collision reports. This illustrates why accuracy alone is insufficient for an imbalanced classification problem.

### Final optimized model comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| **Final Tuned Random Forest** | 0.8462 | 0.2667 | 0.80 | **0.4000** |
| Final Optimized Neural Network | 0.8045 | 0.2267 | 0.85 | 0.3579 |

The final tuned Random Forest identified **16 of 20** collision reports, with **44 false positives** and **4 false negatives**. The final optimized Neural Network identified **17 of 20** collision reports, with **58 false positives** and **3 false negatives**.

I selected the final tuned Random Forest for its stronger F1-score and lower false-positive count in this final comparison. The notebook notes that changes to data preparation mean the **initial and final model scores should not be treated as a strictly like-for-like comparison**.

## Explainability

Random Forest feature importance indicates that **reporting month** and **manufacturer indicators** contribute to the model's decisions. The notebook reports the highest individual feature importance for month, followed by manufacturer indicators including Waymo and Zoox.

These are associations within the reporting data, **not causal explanations** or evidence that one manufacturer is safer than another.

## Key Findings

1. **Class imbalance matters.** High accuracy can hide a failure to identify collision reports.
2. **There is a precision–recall trade-off.** Models that identify more collision reports can also flag more disengagement reports incorrectly.
3. **Hyperparameter tuning does not guarantee improvement.** The best cross-validation setting may not produce the best held-out test result.
4. **Available features are limited.** Manufacturer and month capture reporting patterns, but do not describe the full circumstances surrounding an event.

## Recommendations and Next Steps

- Evaluate additional consistently available information about roadway conditions, weather, operating environment, and vehicle behavior.
- Consider mileage and testing exposure when interpreting differences in reporting frequency.
- Validate any future risk-oriented model on independent data and across reporting periods.
- Avoid using these classification results alone to make operational safety decisions or rank manufacturers.

## Limitations

- The models classify **report type**, not future collision risk.
- The minority collision class is small, making results sensitive to individual observations.
- Manufacturer reporting practices, mileage, and testing environments may differ.
- The model uses only manufacturer and month as common predictors.
- Feature importance is not evidence of causation.

## Repository and Notebook

The complete Jupyter notebook contains data cleaning, EDA, modeling, cross-validation, hyperparameter tuning, evaluation plots, feature importance, and interpretations.

**Notebook:** [AV_Safety.ipynb](./AV_Safety.ipynb)


## Conclusion

Machine learning can identify some patterns distinguishing historical DMV disengagement and collision reports, but the class imbalance and limited shared features make the task challenging. Random Forest models provided the strongest reported F1-scores in the comparisons, while other models sometimes achieved higher collision recall at the cost of more false positives. The results support exploratory analysis of reporting patterns, not claims about future collision probability or manufacturer safety.
