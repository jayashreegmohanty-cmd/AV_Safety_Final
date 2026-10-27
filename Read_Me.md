# Explainable AI for Autonomous Vehicle Safety
## Learning from Disengagement and Crash Reports

**UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence — Capstone Project**

### Associated Notebook

[AV Safety Jupyter Notebook](AV_Safety.ipynb) *(rename the final notebook to `AV_Safety.ipynb` when uploading to GitHub, or update this link)*

## Table of Contents

1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [1. Understanding the Business](#1-understanding-the-business)
4. [2. Understanding the Data](#2-understanding-the-data)
5. [3. Data Preparation](#3-data-preparation)
6. [4. Modeling](#4-modeling)
7. [5. Evaluation](#5-evaluation)
8. [6. Communication and Potential Application](#6-communication-and-potential-application)
9. [Key Findings and Recommendations](#key-findings-and-recommendations)
10. [Limitations and Next Steps](#limitations-and-next-steps)
11. [Conclusion](#conclusion)
12. [References](#references)

# Introduction

I developed this capstone as part of the UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence. I wanted to work on a problem related to autonomous vehicle safety and explore what machine learning could tell us about publicly reported events.

Autonomous vehicles are improving, but disengagements and collisions remain important issues. California's Department of Motor Vehicles (DMV) publishes reports that provide a useful starting point for studying these events. I used these reports to compare machine learning methods and understand the limitations of working with imbalanced safety-related data.

**What this project does:** It classifies historical DMV records as either disengagement reports or collision reports using manufacturer and reporting month. It does **not** predict future crashes or calculate the true safety risk of an autonomous vehicle.

# Methodology

I organized the work using **CRISP-DM (Cross-Industry Standard Process for Data Mining)**, the same overall project framework used in the reference Berkeley capstone. The stages are business understanding, data understanding, data preparation, modeling, evaluation, and communication of results.

# 1. Understanding the Business

## Background and problem

Autonomous vehicle safety is important to manufacturers, regulators, and the public. DMV reports document different types of events, but those reports do not automatically reveal meaningful patterns. I wanted to test whether machine learning could distinguish between the two historical report categories and identify which available inputs were influential.

## Research question

**How effectively can machine learning distinguish historical autonomous vehicle collision reports from disengagement reports, and what can the model results tell us about the available reporting data?**

This is the question directly tested by my completed notebook. A broader goal—predicting future safety events—would require different data and a different study design.

## Project objectives and success criteria

- Prepare and explore the 2024 DMV disengagement and collision reports.
- Train and compare several classification models, starting with a simple baseline.
- Use cross-validation and hyperparameter tuning to evaluate model choices.
- Prioritize **recall and F1-score**, alongside precision and accuracy, because collision records are rare.
- Interpret feature importance carefully and explain what the results can and cannot support.

A model that misses most collision reports would not be useful for this classification task, even if its overall accuracy looks high. At the same time, too many false positives reduce the practical value of a classifier.

# 2. Understanding the Data

## Data sources

The analysis uses public California DMV autonomous vehicle reporting data. The notebook loads disengagement reports, collision reports, and mileage reports for exploratory context. The **classification dataset** combines the two report types for 2024.

| Historical report type | Records | Approximate share |
|---|---:|---:|
| Disengagement | 1,457 | 93.5% |
| Collision | 101 | 6.5% |
| **Total** | **1,558** | **100%** |

This imbalance matters: a model can achieve high accuracy by predicting the more common disengagement category and still fail to identify collisions.

## Exploratory data analysis

I examined reporting patterns across manufacturers and months and used visualizations to understand the distribution of events. I also reviewed mileage information during the exploratory stage, but **mileage is not an input to the final report-type classification model**.

The main observation was that collision reports are much less frequent than disengagement reports. Raw event counts also differ between manufacturers. These counts should not be interpreted as manufacturer safety rankings because testing exposure and reporting circumstances vary.

# 3. Data Preparation

I checked the source datasets for missing values, duplicates, and inconsistent manufacturer names. I standardized manufacturer names and date information, then prepared the two report types for modeling.

I created a binary target variable, **`Risk_Event`**:

- `0`: disengagement report
- `1`: collision report

The main predictors were manufacturer and reporting month. Manufacturer names were encoded for machine learning. The data was split into **80% training and 20% testing**, using a stratified split so the rare collision class remained represented in both sets. The notebook uses `random_state=42` for reproducibility.

For appropriate model configurations, I used class-balancing approaches and feature scaling. The held-out test set used in the final comparison contained **312 records, including 20 collision reports**.

# 4. Modeling

## Baseline model

I began with **Logistic Regression** because it is a straightforward classification method and provides a baseline against which to compare more complex models.

## Model comparisons

I evaluated multiple model types and configurations:

| Model or approach | Reason for including it |
|---|---|
| Logistic Regression | Interpretable baseline |
| Decision Tree | Captures nonlinear splits and is easy to inspect |
| Random Forest | Combines trees and handles nonlinear patterns |
| Gradient Boosting | Tests sequential tree-based learning |
| Neural Network (MLPClassifier) | Compares a neural-network approach |
| Balanced Neural Network | Investigates the effect of emphasizing rare collision reports |
| GridSearchCV Random Forest | Systematic hyperparameter tuning |
| RandomizedSearchCV Random Forest | Alternative hyperparameter search |

The notebook also includes further optimization of the Random Forest and Neural Network models. Cross-validation was used during model development, including five-fold validation.

## Feature importance

The Random Forest feature-importance analysis found that **reporting month** was an important input, along with some manufacturer-related indicators, including Waymo and Zoox categories. This describes how the model separates the historical report types; it does **not** demonstrate that a manufacturer or month causes collisions.

# 5. Evaluation

## Why I used these metrics

- **Accuracy:** The percentage of all reports classified correctly. It can be misleading when one class dominates.
- **Precision:** Among records predicted to be collisions, the fraction that actually are collision reports.
- **Recall:** Among actual collision reports, the fraction the model identifies.
- **F1-score:** A combined measure of precision and recall that helps compare models under class imbalance.

## Initial model comparison

In the initial comparison of eight model configurations, **RandomizedSearchCV Random Forest** had the highest reported test F1-score:

| Metric | Initial RandomizedSearchCV Random Forest |
|---|---:|
| Accuracy | 84.62% |
| Precision | 27.42% |
| Recall | 85.00% |
| F1-score | **0.4146** |
| Collision reports identified | 17 of 20 |

Gradient Boosting and the balanced Neural Network reached **90% recall** in the initial comparison, but at the cost of additional false positives. The original Neural Network achieved **93.59% accuracy** while identifying **no collision records**, illustrating why accuracy alone is insufficient.

**Important:** The notebook states that subsequent cleaning and optimization changes mean the initial and final F1-scores are **not directly comparable as a controlled before-and-after improvement**.

## Final optimized model comparison

| Metric | Tuned Random Forest | Optimized Neural Network |
|---|---:|---:|
| Accuracy | 84.62% | 80.45% |
| Precision | 26.67% | 22.67% |
| Recall | 80.00% | 85.00% |
| F1-score | **0.4000** | 0.3579 |
| True collision reports found | 16 of 20 | 17 of 20 |
| False positives | 44 | 58 |
| False negatives | 4 | 3 |

I selected the **final tuned Random Forest** for its better F1-score and precision among the two final optimized models. The optimized Neural Network identified one more collision report, but it also generated more false positives.

## Interpreting the results

These results show a real trade-off. The tuned Random Forest catches most collision reports, but its precision of 26.67% means many reports flagged as collisions are actually disengagement records. This is not sufficient evidence for a real-world safety-alert system. The small number of collisions in the test set also makes performance estimates sensitive to a few individual records.

# 6. Communication and Potential Application

The outcome of this capstone is a **reproducible analysis and comparison of historical report classification models**, not a deployed product. The results can help explain why class imbalance, feature selection, and careful evaluation matter when working with safety-related datasets.

If this project were expanded, a useful application might be a research dashboard that summarizes reporting patterns and highlights records for further review. Such a tool would need stronger validation and more representative data before being used to support operational safety decisions. No such application was developed or deployed as part of this capstone.

# Key Findings and Recommendations

## Key findings

1. The report categories are highly imbalanced: collision records make up only about 6.5% of the classification data.
2. **Accuracy alone is not a reliable selection metric** for this problem; one high-accuracy Neural Network missed all collision reports.
3. Random Forest configurations offered the strongest observed balance of precision and recall by F1-score in the reported comparisons.
4. Improving recall generally introduced more false positives, so the best choice depends on how the predictions would be used.
5. Manufacturer and month provide limited context. Feature importance helps explain model behavior but does not establish causes or safety rankings.

## Recommendations

For readers analyzing similar datasets, I recommend reporting precision, recall, F1-score, and confusion matrices together rather than relying on accuracy alone. I would also recommend validating models on independent reporting periods and treating manufacturer comparisons cautiously unless differences in mileage and operating conditions can be addressed.

# Limitations and Next Steps

The project has several important limitations:

- It predicts **historical report category**, not the probability of a future collision.
- It uses a small set of predictors—primarily manufacturer and month—rather than detailed road, weather, traffic, or vehicle context.
- There are only **20 collision reports in the held-out test set**.
- Reporting practices and testing exposure can vary across manufacturers.
- Feature importance cannot establish causation.

If I continue the work, I would explore additional data such as miles driven, road conditions, weather, and operating environment. I would also investigate a time-based validation design and a genuinely prospective prediction target before attempting to model future collision risk.

# Conclusion

This capstone gave me practical experience applying machine learning to a real-world, imbalanced dataset. I cleaned and combined public DMV reports, carried out exploratory analysis, compared several classification methods, tuned hyperparameters, and interpreted the results using multiple evaluation metrics.

The strongest lesson was that **a high accuracy score does not necessarily mean a model performs well on the events we care about**. Random Forest gave the best overall F1-score among the reported model comparisons, but its precision also showed the limits of using a small number of features to classify these records.

I see this project as a starting point for further autonomous vehicle safety research rather than a finished collision-risk prediction system.

# References

- California Department of Motor Vehicles (DMV). *Autonomous Vehicle Disengagement Reports* and *Autonomous Vehicle Collision Reports*. https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/
- California DMV. *Autonomous Vehicle Testing Mileage Reports*. California DMV autonomous vehicle reporting materials.
- UC Berkeley Executive Education. *Professional Certificate in Machine Learning and Artificial Intelligence*.
- Scikit-learn documentation. https://scikit-learn.org/stable/

---

**Project source:** The findings and numerical results above are based on the completed `AV_Safety` Jupyter notebook. The organization follows the CRISP-DM presentation style of [Andrés Pineda's Berkeley capstone](https://github.com/andrespineda/UC_Berkeley_MLAI_Capstone), but the analysis and writing describe this autonomous vehicle project.
