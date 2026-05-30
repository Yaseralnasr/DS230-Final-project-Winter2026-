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

