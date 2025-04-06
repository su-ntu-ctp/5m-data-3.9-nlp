# Assignment

## Instructions

### Text Classification for Spam Detection

In this assignment, you will build a text classification model using Naive Bayes to classify SMS messages as spam or ham (non-spam). You will implement text preprocessing techniques and use the Vector Space Model (TF-IDF) to represent the text data.

#### Dataset

You will be using the SMS Spam Collection dataset, which contains a set of SMS messages that have been labeled as either spam or ham (legitimate). This dataset is available through several Python libraries or can be downloaded directly.

#### Tasks

1. **Text Preprocessing**:

   - Load the dataset
   - Implement tokenization
   - Apply stemming or lemmatization
   - Remove stopwords

2. **Feature Extraction**:

   - Use TF-IDF vectorization to convert the text data into numerical features
   - Explore the most important features for spam and ham categories

3. **Classification**:

   - Split the data into training and testing sets
   - Train a Multinomial Naive Bayes classifier
   - Evaluate the model using appropriate metrics (accuracy, precision, recall, F1-score)
   - Create a confusion matrix to visualize the results

4. **Analysis**:
   - Analyze false positives and false negatives
   - Identify characteristics of messages that are frequently misclassified
   - Suggest improvements to your model

#### Starter Code

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import urllib.request

# Load the SMS Spam Collection dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
urllib.request.urlretrieve(url, "sms.tsv")
sms_data = pd.read_csv('sms.tsv', sep='\t')
print(sms_data.head())

# Check data distribution
print(sms_data['label'].value_counts())

# TODO: Implement text preprocessing
# - Tokenization
# - Stemming/Lemmatization
# - Stopwords removal

# TODO: Apply TF-IDF vectorization

# TODO: Split data into training and testing sets

# TODO: Train a Multinomial Naive Bayes classifier

# TODO: Evaluate the model

# TODO: Analyze misclassifications
```

## Submission

- Submit the URL of the GitHub Repository that contains your work to NTU black board.
- Should you reference the work of your classmate(s) or online resources, give them credit by adding either the name of your classmate or URL.
