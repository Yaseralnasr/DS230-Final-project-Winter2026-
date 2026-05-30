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
