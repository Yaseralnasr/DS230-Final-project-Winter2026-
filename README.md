• Project Prepared By:
Heba Al-Naimat,
Yaser Abu Al-Nasr

• Course Instructor:
Dr. Mustafa Ali

Project Overview
This project focuses on predicting the popularity and engagement level of online news articles using machine learning techniques.
The dataset used is the Online News Popularity dataset from the UCI Machine Learning Repository. 
It contains articles published by Mashable over approximately two years and includes various features related to article content, linguistic properties, publication timing, sentiment analysis, keyword statistics, and metadata.
The main objective is to analyze these features and build machine learning models capable of predicting how popular a news article will be based on its characteristics

To achieve this goal, the project was divided into the following sections:
Exploratory Data Analysis (EDA)
Data Preprocessing
Feature Engineering
Classification Models
Classification Model Tuning
Regression Models
Regression Model Tuning
Model Comparison
Learning Curves Analysis

EDA
A comprehensive Exploratory Data Analysis (EDA) was conducted on the Online News Popularity dataset to understand its structure, quality, and the factors influencing article popularity.
The analysis began by loading the dataset and examining its overall structure. The dataset contains approximately 39,000 observations and more than 60 features.
Data quality checks were then performed, including missing value detection and duplicate record analysis. The results showed that no missing values or duplicate rows were present, indicating a clean and reliable dataset.
A statistical summary was generated for all numerical features to examine key descriptive statistics such as mean, standard deviation, minimum, and maximum values. The target variable Shares was analyzed and found to be highly right-skewed with a significant number of outliers. To address this issue, a logarithmic transformation was applied, creating a new variable called log shares.
Outlier handling was performed using percentile-based clipping techniques, while ratio-based features were constrained within valid ranges. Feature distributions were explored through histograms, skewness analysis, IQR-based outlier detection, and boxplots.
Relationships among numerical variables were investigated using a Correlation Heatmap, and the strongest positive and negative correlations with log shares were identified. Additional analyses were conducted to evaluate the impact of several factors on article popularity, including:Day of publicationWeekend versus weekday publishingContent categorySentiment and subjectivity featuresLDA topic distributions
Furthermore, Pair plots and Density Plots were used to explore feature relationships and distribution patterns.Finally, binary popularity labels were created using both Median and 75th Percentile (Q75) thresholds, and class balance was analyzed to support future classification modeling.

# Classification Models Results and Discussion

To predict article popularity, several classification algorithms were trained and evaluated, including Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Linear Support Vector Machine (Linear SVM), and RBF Support Vector Machine (RBF SVM).

After preprocessing the data and applying feature scaling where required, all models were trained using the training dataset and evaluated on the test dataset using multiple performance metrics, including Accuracy, Precision, Recall, F1-Score, Area Under the ROC Curve (AUC), and Matthews Correlation Coefficient (MCC).

## Model Performance Comparison

The comparison results showed that the Random Forest model achieved the best overall performance among all evaluated models. It obtained the highest F1-Score (0.696) and the highest AUC value (0.722), indicating superior classification capability and better discrimination between popular and non-popular articles.

In contrast, the Decision Tree model produced the weakest results, achieving lower Accuracy, F1-Score, and AUC values compared to the other models. Logistic Regression, Linear SVM, and RBF SVM showed relatively similar performance levels, while KNN achieved moderate results.

## ROC Curve Analysis

The ROC curves demonstrated that all classification models performed better than random guessing. However, Random Forest achieved the largest Area Under the Curve (AUC = 0.722), indicating the strongest ability to distinguish between the two classes. Decision Tree achieved the lowest AUC score (0.580), reflecting weaker classification performance.

## Precision–Recall Analysis

The Precision–Recall curves further confirmed the superiority of the Random Forest model. It achieved the highest Average Precision (AP = 0.732), demonstrating a better balance between precision and recall across different classification thresholds. This result indicates that the model is more effective in correctly identifying popular articles while minimizing misclassification.

## Calibration Curve Analysis

Calibration curves were used to evaluate the reliability of the predicted probabilities generated by each model. The results showed that Random Forest, Logistic Regression, and SVM models produced probability estimates closer to the ideal calibration line, suggesting more reliable probability predictions. Other models displayed larger deviations from the ideal calibration curve.

## Matthews Correlation Coefficient (MCC)

To obtain a more balanced evaluation, the Matthews Correlation Coefficient (MCC) was calculated for all models. Random Forest achieved the highest MCC value (0.331), followed closely by RBF SVM (0.323). These results further support the selection of Random Forest as the most balanced and reliable classification model.

## Best Model Evaluation: Random Forest

Based on the previous evaluation metrics, Random Forest was selected as the final model. The confusion matrix showed that:

* 2,211 non-popular articles were correctly classified.
* 1,389 non-popular articles were incorrectly classified as popular.
* 2,926 popular articles were correctly classified.
* 1,167 popular articles were incorrectly classified as non-popular.

The Classification Report revealed the following performance:

| Class       | Precision | Recall | F1-Score |
| ----------- | --------- | ------ | -------- |
| Non-Popular | 0.65      | 0.61   | 0.63     |
| Popular     | 0.68      | 0.71   | 0.70     |

The overall model accuracy reached approximately 67%, demonstrating a satisfactory ability to predict article popularity. The model performed better when identifying popular articles than non-popular articles.

## Learning Curve Analysis

The learning curves were generated to evaluate the model’s learning behavior and generalization ability. The training score remained very close to 1.0 across all training sizes, while the validation score stabilized around 0.69. The noticeable gap between the training and validation curves suggests the presence of moderate overfitting, indicating that the model learns the training data extremely well but does not achieve the same level of performance on unseen data.

## Conclusion

Overall, the Random Forest classifier outperformed all other evaluated models across multiple performance metrics, including F1-Score, AUC, and MCC. The model demonstrated a strong ability to predict article popularity and achieved the most balanced classification performance. Although some evidence of overfitting was observed, Random Forest remained the most suitable model for this study and was selected as the final predictive model.

Overall, the EDA revealed a high-quality dataset with meaningful patterns and relationships, providing a strong foundation for building machine learning models to predict online news popularity.


# Data Preprocessing and Exploratory Data Analysis

## 1. Dataset Overview

The dataset was initially loaded and inspected to understand its structure and characteristics. The original dataset contained **39,644 observations** and **59 features**. Data type analysis showed that most variables were numerical, and memory usage was approximately **18.71 MB**, indicating that the dataset could be processed efficiently without additional optimization.

## 2. Missing Values and Duplicate Records

A comprehensive data quality assessment was performed to identify missing and duplicate values. The analysis confirmed that:

* No missing values were found in any feature.
* No duplicate records were detected.

These findings indicate that the dataset was already highly complete and consistent.

## 3. Target Variable Analysis

The target variable, **shares**, was analyzed to understand its distribution. The results showed a highly right-skewed distribution with:

* Mean = 3395
* Median = 1400
* Skewness = 33.96

To reduce skewness and improve model performance, a logarithmic transformation was applied, creating the variable **log_shares**. A binary classification target (**popular**) was then generated using a threshold of 1400 shares. The resulting class distribution was relatively balanced:

* Not Popular: 51.1%
* Popular: 48.9%

This balance makes the dataset suitable for classification tasks.

## 4. Feature Distribution Analysis

Histograms and density plots were generated for all numerical variables. The analysis revealed that most features exhibited strong positive skewness, with values concentrated near zero. A smaller subset of features, particularly sentiment-related variables, showed distributions closer to normality.

These findings suggested that some variables would benefit from transformation or outlier treatment before model development.

## 5. Correlation and Relationship Analysis

Correlation analysis was conducted to identify relationships between features and the target variable. The heatmap indicated generally weak correlations with article popularity.

The strongest observed correlation with log(shares) was:

* kw_avg_avg (r ≈ 0.22)

Scatter plots further confirmed that no single feature had a strong linear relationship with popularity, indicating that predictive performance would likely depend on combining multiple features rather than relying on individual variables.

## 6. Publishing Day Analysis

The effect of publication day on article popularity was examined.

The results showed that:

* Saturday achieved the highest median number of shares (approximately 1,900).
* Sunday produced similar high performance.
* Weekday articles consistently generated lower engagement levels.

This suggests that publication timing may influence article popularity.

## 7. Content Channel Analysis

Popularity was analyzed across different content categories.

The findings showed that:

* Social Media and Technology articles achieved the highest popularity levels.
* Lifestyle content also performed relatively well.
* Entertainment and World categories produced the lowest engagement levels.

These results indicate that content category plays a meaningful role in article performance.

## 8. Sentiment Analysis

The relationship between sentiment features and article popularity was investigated using scatter plots and boxplots.

The analysis revealed that:

* Global subjectivity showed no clear relationship with popularity.
* Sentiment polarity had only a minor impact on the number of shares.
* Median share counts remained relatively stable across sentiment categories.

Therefore, sentiment features appeared to have limited predictive power compared with other content-related variables.

## 9. Data Quality Validation

A search for impossible and suspicious values was performed.

The results identified:

* 1,182 articles (approximately 2.98%) with zero content tokens.

These records represent empty or invalid articles and were removed from the dataset.

No other significant suspicious values were detected.

## 10. Data Cleaning

After removing invalid observations, the dataset size changed from:

* Before cleaning: 39,644 rows
* After cleaning: 38,462 rows

All remaining records passed the data quality checks successfully.

## 11. Outlier Detection

Outliers were identified using the Interquartile Range (IQR) method.

The analysis showed substantial outlier percentages in several variables, including:

* kw_max_max (24.35%)
* title_sentiment_polarity (20.59%)
* num_imgs (19.41%)

These results confirmed the presence of extreme values that could potentially influence model performance.

## 12. Outlier Treatment

Two approaches were evaluated:

1. Outlier Removal
2. Winsorization

Outlier removal significantly reduced the number of observations, while Winsorization preserved all records by capping extreme values at selected percentile thresholds.

Because it retained all available information while reducing the influence of extreme observations, Winsorization was selected as the preferred method.

The technique was applied to all continuous numerical variables using the 1st and 99th percentile limits.

## 13. Feature Scaling

Several scaling techniques were evaluated, including:

* StandardScaler
* MinMaxScaler
* RobustScaler

The comparison showed that scaling changed feature ranges but did not alter the underlying skewed distributions. Based on model requirements:

* Logistic Regression, SVM, and KNN require scaling.
* Decision Tree, Random Forest, and XGBoost do not require scaling.

Appropriate scaling methods were selected accordingly.

## 14. Feature Type Verification

Feature types were reviewed after preprocessing.

The final dataset contained:

* 15 binary variables
* 29 continuous variables
* 0 object/text variables

All categorical information had been successfully encoded into numerical form.

## 15. Final Dataset

The cleaned and preprocessed dataset was saved for machine learning experiments.

Final dataset characteristics:

* Rows: 38,462
* Columns: 61
* Target variables:

  * shares
  * log_shares
  * popular

The dataset was successfully prepared for classification and predictive modeling tasks.


Feature Engineering and Feature Selection
1. Feature Engineering

To improve the predictive power of the dataset, several new features were generated from the original variables. The objective was to capture hidden relationships, reduce skewness, and create more informative representations of article characteristics.

1.1 Keyword-Based Features

Several features were created to measure keyword density and keyword interactions:

keyword_density
kw_avg_min
kw_avg_max
kw_avg_avg

These variables combine keyword statistics and article content information to better represent topic relevance and content richness.

1.2 Sentiment Features

Additional sentiment-related variables were generated, including:

sentiment_score
polarity_ratio

These features summarize the emotional characteristics of articles and provide a more compact representation of sentiment information.

1.3 Temporal Features

New time-related features were generated, such as:

is_weekend
weekday_is_saturday
weekday_is_sunday

These variables were created to capture the effect of publication timing on article popularity.

1.4 Logarithmic Transformations

Because many variables showed highly skewed distributions, logarithmic transformations were applied to selected features, including:

log_num_imgs
log_num_videos
log_kw_avg_avg

The transformation reduced skewness and improved feature distributions, making them more suitable for machine learning algorithms.

2. Analysis of Engineered Features

Histograms and scatter plots were used to examine the distributions of the newly created features and their relationships with the target variable.

The analysis revealed that:

Most engineered variables remained right-skewed but showed improved distributions after logarithmic transformation.
Keyword-related features demonstrated stronger relationships with article popularity than many original variables.
Sentiment-based features showed moderate predictive potential.
Weekend publication indicators appeared to influence article popularity.

After feature engineering, the total number of features increased from 59 to 75 features, resulting in a richer representation of the dataset.

3. Feature Selection

Three different feature selection techniques were applied to identify the most relevant predictors.

3.1 Correlation-Based Feature Selection

Pearson correlation coefficients were calculated between each feature and the target variable (popular).

Features with absolute correlation greater than 0.05 were retained.

Results:

Selected Features: 37
Most correlated variables included:
log_kw_avg_avg
kw_avg_avg
LDA_02
data_channel_is_world
is_weekend
kw_max_avg
global_subjectivity

The highest correlation was observed for log_kw_avg_avg (approximately 0.20).

3.2 Random Forest Feature Importance

A Random Forest classifier was trained to evaluate feature importance based on ensemble tree learning.

Results:

Selected Features: 23
Most important variables:
log_kw_avg_avg
log_kw_min_avg
log_kw_avg_avg
is_weekend
LDA_02
kw_min_avg
data_channel_is_entertainment

The Random Forest method highlighted several non-linear relationships that were not fully captured by correlation analysis.

3.3 Recursive Feature Elimination (RFE)

Recursive Feature Elimination was performed using Logistic Regression as the base estimator.

Results:

Selected Features: 30
Top-ranked variables:
log_kw_avg_avg
kw_num_avg
log_kw_min_content
weekday_is_sunday
global_subjectivity
keyword_density

RFE successfully identified features that contributed most to classification performance while removing redundant variables.

4. Comparison of Feature Selection Methods

The number of selected features obtained from each method is summarized below:

Method	Selected Features
Correlation Analysis	37
Random Forest	23
RFE	30

Only 15 features were selected by all three methods simultaneously.

The common features included:

log_kw_avg_avg
LDA_02
LDA_04
is_weekend
data_channel_is_entertainment
data_channel_is_tech
global_subjectivity
keyword_density
kw_min_avg
kw_avg_avg

These features were considered the most reliable predictors of article popularity.

5. Principal Component Analysis (PCA)

PCA was performed to investigate the intrinsic dimensionality of the dataset.

Results:

The first two principal components explained approximately 22.95% of the total variance.
Approximately 30 components were required to explain 90% of the variance.
Approximately 36 components were required to explain 95% of the variance.

These findings indicate that article popularity depends on multiple interacting factors rather than a small number of dominant variables.

6. Final Feature Set

Based on the combined results of Correlation Analysis, Random Forest, and RFE, a final set of 15 highly informative features was selected.

Top features included:

log_kw_avg_avg
kw_avg_avg
LDA_02
is_weekend
kw_min_avg
kw_subjectivity
data_channel_is_socmed
data_channel_is_entertainment
weekday_is_saturday
global_subjectivity
data_channel_is_tech
log_kw_min_avg
log_kw_max_avg
LDA_01
LDA_00

The final engineered dataset contained 39,642 observations and 81 columns, including the target variable, and was prepared for the model-building phase.

