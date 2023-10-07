# Twitter-Airline-Sentiment-Analysis



# Sentiment Analysis:
This repository contains the analysis of a Sentiment Analysis dataset. The analysis aims to predict sentiment labels for tweets.

# Dataset:
The dataset used for the analysis contains information on various features such as tweet text, airline, and user-related details. The target variable is the airline_sentiment, which represents the sentiment of the tweet (positive, negative, or neutral).

# Analysis Steps:

**Data Exploration:** Perform exploratory data analysis to gain insights into the dataset, identify patterns, and understand the distributions and relationships between variables.

**Data Preprocessing:** Clean the data by handling missing values, encoding categorical variables, and preparing the text data for analysis.

**Text Tokenization:** Tokenize the text data to break down the tweets into individual words or tokens.

**Text Vectorization:** Convert the tokenized text data into numerical form using techniques like TF-IDF vectorization.

**Model Building:** Train and evaluate various machine learning models, such as Logistic Regression, Random Forest, Naive Bayes, etc., to predict sentiment labels.

**Model Evaluation:** Assess the performance of the models using evaluation metrics like accuracy, precision, recall, and F1-score.

**Hyperparameter Tuning:** Fine-tune the model hyperparameters using techniques like Grid Search or Random Search to improve model performance.

**Conclusion:** Summarize the key findings from the analysis, including the most accurate model and insights into sentiment patterns in the Twitter data.

# Results

**Logistic Regression:**

Initial Model: 

   - Accuracy 69.5% 
   - Recall Score 61.3%

After Hyperparameter Tuning:

Accuracy 73.3%, 
Recall Score 64.1%

**Decision Tree Classification:**

Accuracy 68.5%
Recall Score 58.1%

**RandomForest Classification:**

Initial Model: 

Accuracy 76.5%
Recall Score 64.2%

After Hyperparameter Tuning: 

Accuracy 77.0%
Recall Score 64.9%

**Support Vector Classification:**

Accuracy 74.6%
Recall Score 59.6%

**K-Nearest Neighbors (KNN):**

Accuracy 31.8%
Recall Score 46.4%

**XGBoost Classification:**

Accuracy 76.3%
Recall Score 65.0%

These results indicate that after hyperparameter tuning, the Logistic Regression model achieved the highest accuracy of 73.3% and a recall score of 64.1%. 
However, it's important to note that different models may have varying strengths and weaknesses, and the choice of the best model depends on the specific goals and trade-offs of the analysis.
