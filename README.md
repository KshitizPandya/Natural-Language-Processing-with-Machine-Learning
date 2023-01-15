![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
# Natural Language Processing with Machine Learning


This repository covers the basic understanding of what is **Natural Language Processing (NLP)** and how we can achieve it using **Natural Language Tool Kit (NLTK**).

We have used dataset given named as `SMSSpamCollection.tsv` which is a tab-seperated-values file. Our dataset is a semistructured text data of various SMSs which are marked as ham or spam. 


Note - While reading this documentation, you can directly click on the topic hyperlinked to 
its respective `.ipynb` files.

[Click Here](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning), to directly refer to the repository.

[Click Here](https://drive.google.com/file/d/1QIDyNzIfTYHR92-sQlRys_gpycBKnkLG/view?usp=sharing), to download the dataset directly.


<!-- ![2nd](https://user-images.githubusercontent.com/110394695/212464339-aec410c0-01e1-40e5-9982-98bb8aab189a.gif) -->
<img src="https://user-images.githubusercontent.com/110394695/212464339-aec410c0-01e1-40e5-9982-98bb8aab189a.gif" width="800" height="100">


## Table of Contents
* [NLP Basics](#NLP-Basics)
* [Data Cleaning](#Data-Cleaning)
* [Vectorizing Raw Data](#Vectorizing-Raw-Data)
* [Feature Engineering](#Feature-Engineering)
* [Building Machine Learning Classifiers](#Building-Machine-Learning-Classifiers)

## *NLP Basics*
*According to IBM - 
**Natural Language Processing (NLP)** refers to the branch of computer science and more specifically, the branch of Artificial Intelligence concerned with giving computers the ability to understnad text and spoken words in much the same way human beings can.*

### [What is NLP](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/1.%20NLP%20Basics/1.1.%20what%20is%20NLP.ipynb)
First step of any project installation of necessary modules. Here we give light how to install NLTK and explore different functions of nltk module.

### [Reading in Text Data & Why Do We Need to Clean the Text](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/1.%20NLP%20Basics/1.2.%20reading%20in%20text%20data%20%26%20why%20do%20we%20need%20cleaning.ipynb)
The first step of any task performed for processing is reading the data and retriving the information.

Points covered here are:
* splitting the data and create a list.
* differentiating labels and texts.

### [How to Explore a Dataset](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/1.%20NLP%20Basics/1.3.%20How%20to%20explore%20a%20dataset.ipynb)
Exploring the provided dataset is the next step to working with any big project which 
requires NLP. 

Points covered here are: 
* reading in the text data from dataset (Here - `SMSSpamCollection.tsv`).
* Exploring the dataset - shape, how many spam/ham are there?, how much missing data is there i.e, null values?

### [Learning How to Use Regular Expressions](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/1.%20NLP%20Basics/1.4.%20learning%20how%20to%20use%20regular%20expressions.ipynb)
Regular expressions or in short known as regex is a defined as a sequence of characters 
that specificies pattern in text.

Points covered here are:
* What is `re package`.
* difference between `.split()` and `.findall()`
* splitting a sentence into a list of words.
    - exploring - \s, \s+, \W+, \S+, and various parameters.
* replacing a specific string.

Other functions of 're' can also be explored such as .search(), .match(), etc..

### [Implementating a Pipeline to Clean the Text](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/1.%20NLP%20Basics/1.5.%20implementing%20a%20pipeline%20to%20clean%20text.ipynb)
Implementation of pre build pipelins for cleaning the text is a necessary task to
get the desired output more efficiently. 

Points covered here are: 
* Remove Punctuations
* Tokenization
* Removing stopwords

## *Data Cleaning*
***Data cleaning** is the process of fixing or removing incorrect, corrupted, incorrectly formatted, duplicate, or incomplete data within
a dataset. Removing duplicates is one of the main concern of data cleaning.*

To perform data cleaning we explore several apporach here.

### [Stemming](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/2.%20Data%20Cleaning/2.1.%20stemming.ipynb)
Stemming is important in **Natural Language Understanding (NLU)** and **Natural Language Processing (NLP)**.

Stemming is the process of reducing a word to its stem that affixes to suffixes and prefixes 
or to the roots of words known as "lemmas".

*In simple words - Stemming can be defined as a process of reducing 
dervied words to their word stem or root word.*

For example
- grows, growing, grow - when working with NLP models, in many situations it seems as if it would be useful for a search for one of these words to return which should be root to other similar words, here - 'grow'.
- run, running, runner - here it seems that the root word is same which is - 'run', but this consideration is wrong as run and running are tasks but runner signifies a person. This should be clear to the model in its learning phase to avoid any misinterpretations or unsatisfactory outputs.

To perform this task, `stemming` is used.

There are several types of stemmers available:
- Porter stemmer
- Snowball stemmer
- Lancaster stemmer
- Regex-based stemmer

Our focus here will be on `Porter Stemmer`, that how to use it in our dataset and create our dataframe to work further with it.

### [Lemmatizing](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/2.%20Data%20Cleaning/2.2.%20lemmatizing.ipynb)
Lemmatization usually refers to doing things properly with the use of a **vocabulary** and **morphological** analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma.

*In simple words - Lemmatization is a process of grouping together the inflected word so they can be analyzed as a single term.*

Then, What is the difference between Stemming and Lemmatization?

Let's explore it with an example - 

- meanness, meaning - these two words way different from each other in sort of their actual meaning. When the input of these two words is given to Stemmer, it gives the output as 'mean' for both of them, which basically is wrong. But, when the same input of both words is given to lemmatizers, it gives the output meanness and meaning only. Why so? It is because lemmatizers focuses vocabulary as well as dictionary form and meaning of the word giving us the more satisfactory output. 
- similarly in case of goose and geese. Root word is same, as goose is singular whereas the latter is plural. Stemmer gives output for both as 'goos and gees' which is obviously wrong as it is not even a word. But interstingly, lemmatizer gives output as 'goose' understanding that both the words refer to the singular word goose.

There are several lemmatizer present as well, but here we will use `WordNet Lemmatizer`, and see how it responds to out dataset.

## *Vectorizing Raw Data*
*Basically, **Vectorization** is a jargon for classic approach of converting input data from its raw format (i.e, text) into vectors of real numbers which is the format that ML model support.*

To achieve this task we focus on vectorizing the dataset here.

### [Count Vectorization](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/3.%20Vectorizing%20Raw%20Data/3.1.%20count%20vectoriztion.ipynb)
NLP models can't understand Textual data. They only accept **numbers**. So, this
textual data needs to converted to numbers. The process of achieving this task is known as **Vectorixation**.

Vectorization can also be explained as a process of encoding text as integers to create 
**feature vectors** which is an n-dimensional vector of numerical features that represent
some object.

In simple words, our **Count Vectorizer** creates a document-term matrix where the entry of each cell will be 
a count of the number of times that word occurred in that document.

For example, we have a sentence: 

`My Name Is XYZ. Firstly, I Completed My  B.E. In 2019 From Gujarat Technology University. I Like Playing Cricket And Reading Books. Also, I Am From Amreli Which Is Located In Gujrat.`

Its corresponding output sparse matrix after vectorization will be something like this:

![output_1](https://user-images.githubusercontent.com/110394695/211851413-df2c3a69-3cd4-4558-ae96-bf86c276f10f.png)
![output_2](https://user-images.githubusercontent.com/110394695/211851464-b1ce397d-6f9e-4889-a0c0-4ccd4b623ef4.png)


### [N Grams](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/3.%20Vectorizing%20Raw%20Data/3.2.%20N_grams.ipynb)
N-grams are continuous sequences of words or symbols or tokens in a document. In Techincal terms, they can be defined as the neighbouring sequences of items in a document.

**N-grams** creates a document term matrix where counts still occupy the cell
but instead of the columns representing single terms, they represent all combinations of adjacent words of length n in your text.

In simple words, if we take an example into consideration of a sentence:

`NLP is an intersting topic`

it can be divided into parts easily, like bigrams, trigrams, fourgrams, etc., which signifies sequence of two words, three words, and four words respectively.

| n | Name      | Tokens                                                         |
|---|-----------|----------------------------------------------------------------|
| 2 | bigram    | ["nlp is", "is an", "an interesting", "interesting topic"]      |
| 3 | trigram   | ["nlp is an", "is an interesting", "an interesting topic"] |
| 4 | four-gram | ["nlp is an interesting", "is an interesting topic"]    |


### [TF-IDF](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/3.%20Vectorizing%20Raw%20Data/3.3.%20TF-IDF.ipynb)
TF-IDF stands for, **Term Frequency - Inverse Document Frequency**.

TF-IDF (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.

This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.

TF-IDF for a word in a document is calculated by multiplying two different metrics:

* The term frequency of a word in a document. There are several ways of calculating this frequency, with the simplest being a raw count of instances a word appears in a document. Then, there are ways to adjust the frequency, by length of a document, or by the raw frequency of the most frequent word in a document.
* The inverse document frequency of the word across a set of documents. This means, how common or rare a word is in the entire document set. The closer it is to 0, the more common a word is. This metric can be calculated by taking the total number of documents, dividing it by the number of documents that contain a word, and calculating the logarithm.
* So, if the word is very common and appears in many documents, this number will approach 0. Otherwise, it will approach 1.

To put this in mathematical form, it can be can calculated using the given equation:

![tf-idf equation](https://user-images.githubusercontent.com/110394695/211858157-8d28c363-fc93-458e-b237-78bc680aa3c2.png)

## *Feature Engineering*
Feature engineering is the pre-processing step of machine learning, which is used to transform raw data into features that can be used for creating a predictive model using Machine learning or statistical Modelling. 
Simply putting it, feature engineering aims to imrpove the performance of the models.

### [Feature Creation](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/4.%20Feature%20Engineering/4.1.%20Feature%20Creation.ipynb)
Feature creation is the process of taking raw data and transforming it into features that can be used in machine learning models. This includes extracting relevant information, identifying specific keywords, calculating summary statistics, converting categorical variables into numerical variables, combining multiple features, etc. These resulting features are used as input for the model, allowing it to make predictions or classifications based on the information contained in the features.

![feature-engineering-for-machine-learning2](https://user-images.githubusercontent.com/110394695/212463269-8994797d-4a99-4296-ac28-2dba434ce792.png)


Some examples of feature creation include:

* Extracting the length of text in a document as a feature for a text classification model.

* Identifying the presence of specific keywords in customer reviews as features for a sentiment analysis model.

* Calculating the average color of an image as a feature for an image classification model.

* Converting categorical variables into numerical variables by using one-hot encoding for a model that will be used for prediction or classification.

* Extracting the frequency of certain words in a text for text classification or natural language processing tasks.

* Combining multiple features such as age, income, and education level to create a new feature for a model that will be used for predicting customer behavior.



### [Feature Transformation](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/4.%20Feature%20Engineering/4.2.%20Transformation.ipynb)
In machine learning, feature creation refers to the process of taking raw data and transforming it into features that can be used as input for a model. The goal of feature creation is to extract the most relevant information from the raw data and represent it in a format that is useful for the model. This process is also known as feature engineering or feature extraction.
There are a variety of techniques that can be used for feature creation, depending on the type of data and the machine learning task. Some examples include:

* Extracting specific information from raw data: For example, in a text classification task, the raw data may be a set of documents, and the features could be the frequency of specific words or phrases in those documents.

* Creating new features by combining or transforming existing ones: For example, in a dataset with multiple features such as age, income, and education level, a new feature could be created by combining these features in a meaningful way, such as a "socioeconomic status" feature.

* Encoding categorical variables: Some machine learning models can only handle numerical data, so categorical variables (such as "red", "green", "blue") must be encoded as numerical values (such as 0, 1, 2).

**Types of Transformation**:

![feature-transformation-in-data-mining](https://user-images.githubusercontent.com/110394695/212464857-face8fab-a7ae-4d56-96d9-167e86f1cafe.png)

Here, we will learn how to do *Box-cow Power Tranformation*

Where the equation's base form is: y<sup>x</sup>

Transformation follows the process:
* Determine what range of exponents to test.
* Apply each transformation to each value of your chosen feature.
* Use some criteria to determine which of the transformations yield the best distribution.

| X    | Base Form                | Transformation           |
|------|--------------------------|--------------------------|
| -2   | y<sup>-2</sup>           | 1/y<sup>2</sup>          |
| -1   | y<sup>-1</sup>           | 1/y                      |
| -0.5 | y<sup>-1/2</sup>         | 1/ $\sqrt{y}$            |
| 0    | y<sup>0</sup>            | log(y)                   |
| 0.5  | y<sup>1/2</sup>          | $\sqrt{y}$               |
| 1    | y<sup>1</sup>            | y                        |
| 2    | y<sup>2</sup>            | y<sup>2</sup>            |


## *Building Machine Learning Clssifiers*
Now, as we are this far, we will start building our ML model. 

So, what is a classifier in Machine Learning?

A classifier in machine learning is an algorithm that automatically orders or categorizes data into one or more set of "classes". 

Let's take a very common example, i.e., E-mail. An E-mail classifier always scans the mails recieved and marks them as class labels: `spam mails` or `non spam mails`. 

Then the question arises, what is the difference between a ML classifier and an ML model?

Well, a classifier is the algorithm itself - the rules used by machines to classify data. A model on the other hand, is the end result of your classifier's machine learning.

*The model is trained using the classfier, so that the model, ultimately **classifes** your data*.

The topics below are the direct implementation and knowledge on how to perform the tasks of classifying and selecting the best model suitable. Therefore, let's deep dive into the key concepts performed below before we implement it.


#### **What is an ensemble method?** 
Technique that creates multiple models and then combines them to produce better results than any of the single models individually.
#### **What is a Random Forest Model?** 
Random Forest is a powerful machine learning algorithm that **creates multiple decision trees and combines them to produce a more accurate and stable prediction**. It is widely used in various applications, such as image and speech recognition, natural language processing and bioinformatics.

Let's do a bit more deep diving into Random forest. 

Random Forest is a popular ensemble machine learning algorithm that is based on decision trees. It is a type of **supervised learning** algorithm that can be used for both classification and regression tasks.

A decision tree is a flowchart-like tree structure where an internal node represents a feature(or attribute), the branch represents a decision rule, and each leaf node represents the outcome. In a random forest, multiple decision trees are created, and the final output is determined by the majority vote of the decision trees.

Random Forest uses a technique called bootstrapping, which involves randomly selecting a subset of data from the original dataset to train each decision tree. This helps to reduce overfitting, which is a common problem with decision trees.

It also uses a technique called random feature selection, which involves randomly selecting a subset of features to split on at each internal node of the decision tree. This helps to decorrelate the trees and improve the overall performance of the model.

![random_forest-removebg-preview](https://user-images.githubusercontent.com/110394695/212522727-c1fd4757-9e07-443d-a734-6fc41026feca.png)

Random Forest models have several advantages over traditional decision tree models, including:

* They are less prone to overfitting, as a large number of relatively uncorrelated trees are combined.
* They are more accurate and robust than single decision trees.
* They can handle missing values and large number of categorical variables.
* They can give feature importance which is useful in feature selection.

#### **What is a Gradient Boosting Model?**
Gradient Boosting is a powerful machine learning algorithm that creates many decision trees in a sequential manner to improve the accuracy of the model. It has been widely adopted in various fields, such as computer vision, natural language processing and bioinformatics, and it is implemented in popular libraries like XGBoost, LightGBM, and CatBoost.

Let's deep dive into it. 

Gradient Boosting is a type of ensemble machine learning algorithm that is used for both classification and regression tasks. It is similar to Random Forest, but it creates the decision trees one at a time in a sequential manner and each new tree is created to correct the errors made by the previous tree.

The algorithm starts by fitting a simple model, such as a decision tree, to the data. Then, it iteratively adds new decision trees to the model, each tree trying to correct the errors made by the previous trees. The idea behind this is that by combining many weak learners (i.e. decision trees with high bias and low variance), a strong learner (i.e. a model with low bias and high variance) can be created.

One of the key features of gradient boosting is the use of gradient descent to minimize the loss function, which measures the difference between the predicted values and the actual values. The algorithm adjusts the parameters of the decision trees in order to minimize this loss function, hence the name "gradient boosting."

![The-architecture-of-Gradient-Boosting-Decision-Tree-removebg-preview](https://user-images.githubusercontent.com/110394695/212522826-cb3b6acf-21a4-4091-8b1f-3ce9fd53820c.png)

Gradient Boosting has several advantages over traditional decision tree models, including:

* It can handle missing values and categorical variables
* It can work well with large datasets
* It can improve the performance of the model by iteratively adding decision trees
* It can give feature importance which is useful in feature selection


### [Building a Basic Random Forest Model](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/5.%20Building%20Machine%20Learning%20Classifiers/5.1.%20Building%20a%20basic%20Random%20Forest%20Model.ipynb)
Here, we explored how to build a random forest model.

Exploring of RandomForestClassifiers attributes and hyperparameters and taken into consideration.

Also, light has been given to exploring RandomForestClassifier via Cross - Validation. 

#### **What is Cross - Validation?**
Cross-validation is a technique used to evaluate the performance of a machine learning model by dividing the data into two or more partitions: a training set and one or more validation sets. The model is trained on the training set and then evaluated on the validation set(s).

There are different types of cross-validation methods, but the most common ones are:

* **K-fold cross-validation:** The data is divided into k partitions, where k-1 partitions are used for training, and the remaining partition is used for validation. This process is repeated k times, with each partition being used as the validation set once. The final performance score is the average of all k iterations.

* **Leave-p-out cross-validation:** p data points are left out as the validation set and the remaining data is used for training. This process is repeated for all possible combinations of p data points.

* **Holdout cross-validation:** A fixed percentage of the data is used as the validation set and the remaining data is used for training. This process is repeated multiple times with different random subsets of data being used as the validation set.

Cross-validation is a powerful technique that helps to mitigate the problem of overfitting. It allows the model to be evaluated on unseen data and provides a more accurate estimate of the model's performance on new data. It is also useful for model selection as it allows comparing the performance of different models and choose the best one.

It's important to keep in mind that cross-validation should be performed after the feature creation process and should be done on the same dataset that will be used for the final evaluation, to ensure that the model generalizes well to new unseen data.

NOTE: Here, we explored K-fold cross validation.
### [Random Forest on a Holdout Test Set](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/5.%20Building%20Machine%20Learning%20Classifiers/5.2.%20Random%20Forest%20on%20a%20holdout%20test%20set.ipynb)
Hold out test set can be simply explained as a sample of data not used in fitting a model for the purpose of evaluating the model's ability to generalize unseen data.

Therefore, here we used this concept to gain the basic understanding on it works.

Before going further, it is important to know what is classification evaluation parameters: Accuracy, Precision and Recall.

**Accuracy-** it is defined as the number of true positives and true negatives divided by the number of true positives, true negatives, false positives, and false negatives.

Accuracy = ${Number \space of \space correct \space predictions} \over {Total \space number \space of \space obeseravations}$

**Precision-** Precision is defined as the ratio of correctly classified positive samples (True Positive) to a total number of classified positive samples (either correctly or incorrectly).

Precision = ${Number \space of \space prediction \space as \space spam \space that \space are \space actually \space spam} \over {Total \space number \space of \space prediction \space as \space spam}$

**Recall-** The recall is calculated as the ratio between the numbers of Positive samples correctly classified as Positive to the total number of Positive samples. The recall measures the model's ability to detect positive samples. The higher the recall, the more positive samples detected. 

Recall = ${Number \space of \space prediction \space as \space spam \space that \space are \space actually \space spam} \over {Total \space number \space of \space predictions \space that \space are \space actually \space spam}$

### [Explore Random Forest Model with Grid Search](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/5.%20Building%20Machine%20Learning%20Classifiers/5.3.%20Explore%20Random%20Forest%20Model%20with%20Grid-Search.ipynb)
Grid search is a technique used to find the best set of hyperparameters for a machine learning model. Hyperparameters are parameters that are not learned from the data but are set prior to training the model. Examples of hyperparameters include the learning rate, the number of trees in a random forest, or the regularization strength in a linear regression model.

In grid search, a set of possible values for each hyperparameter is defined, and the algorithm will train and evaluate the model for each combination of hyperparameter values. This process creates a "grid" of possible hyperparameter combinations, hence the name "grid search".

For example, if we want to find the best number of trees and the best maximum depth for a random forest model, we could set the range of values for the number of trees from 20 to 100 with a step of 10, and for the maximum depth from 2 to 8 with a step of 2. This would create a grid of (8*5=40) possible combinations, and the algorithm will train and evaluate the model for each combination.

The grid search process will return the combination of hyperparameters that performed the best on the validation set. The model can then be retrained using these optimal hyperparameters on the entire training set, and then evaluated on the test set.

Therefore where the output of normal models looks like this: 
![normal](https://user-images.githubusercontent.com/110394695/212532422-81d56543-eb90-4f1d-884a-6165c5abd864.png)

GridSearch model's evaluation looks like this:
![grid search](https://user-images.githubusercontent.com/110394695/212532439-025d3410-261f-431e-81f4-7d166f4fa9ba.png)



### [Evaluate Random Forest with GridSearchCV](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/5.%20Building%20Machine%20Learning%20Classifiers/5.4.%20Evaluate%20Random%20Forest%20with%20GridSearchCV.ipynb)
GridSearchCV is a scikit-learn class that implements grid search with cross-validation. It combines the grid search technique with cross-validation to find the best set of hyperparameters for a machine learning model. It takes as input a model, a set of possible values for each hyperparameter, and a scoring metric, and it returns the best set of hyperparameters that maximize the scoring metric.

The class performs k-fold cross-validation for each combination of hyperparameters, where k is a user-specified number. It then returns the combination of hyperparameters that performed the best on the validation set.

For example, if we want to find the best number of trees and the best maximum depth for a random forest model, we could use the GridSearchCV class as follows:

```python

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {'n_estimators': [20, 50, 100], 'max_depth': [2, 4, 8]}
rf = RandomForestClassifier()
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Now you can access the best parameters and the best score using the following properties:
print("Best parameters: ", grid_search.best_params_)
print("Best accuracy: ", grid_search.best_score_)

# you can also access the best estimator by using:
best_rf = grid_search.best_estimator_

#You can also check all the results using
results = grid_search.cv_results_

#you can also use predict function on the best estimator
y_pred = best_rf.predict(X_test)

# you can also calculate accuracy for the best estimator
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: ", accuracy)

```

### [Explore Gradient Boosting Model with Grid-Search](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/5.%20Building%20Machine%20Learning%20Classifiers/5.5.%20Explore%20Gradient%20Boosting%20model%20with%20Grid-Search.ipynb)
What grid searching is, is already explained in the above section while exploring the Random forest model.

Therefore, here diving straight on to point, we explored what are the gradient boosting classifiers, and how to build our own grid-search model.

For curiosity, the output of gradient boosting looks something like this (NOTE - Formatting depends personally in your ways):
![gradient boosting output](https://user-images.githubusercontent.com/110394695/212533400-994b3eb0-c77a-4389-9b60-97c1265cead6.png)


### [Evaluate Gradient Boosting with GridSearchCV](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/5.%20Building%20Machine%20Learning%20Classifiers/5.6.%20Evaluate%20Gradient%20Boosting%20with%20GridSearchCV.ipynb)
GridSearchCV is explained in the above sections already. 

Here we learned how actually implement GridSearchCV for gradient boosting and getting our necessary outputs.

Taking the above example into consideration again, gradient boosting can be achieved like this:

```python

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {'n_estimators': [20, 50, 100], 'max_depth': [2, 4, 8]}
gb = RandomForestClassifier()
grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Now you can access the best parameters and the best score using the following properties:
print("Best parameters: ", grid_search.best_params_)
print("Best accuracy: ", grid_search.best_score_)

# you can also access the best estimator by using:
best_rf = grid_search.best_estimator_

#You can also check all the results using
results = grid_search.cv_results_

#you can also use predict function on the best estimator
y_pred = best_rf.predict(X_test)

# you can also calculate accuracy for the best estimator
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: ", accuracy)

```
### [Model Selection](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/5.%20Building%20Machine%20Learning%20Classifiers/5.7.%20Model%20Selection.ipynb)
To evaluate which model is the best fit for a task, you should compare the accuracy, precision, and recall of both the Random Forest and Gradient Boosting models on the same dataset.

If the goal of the task is to maximize accuracy, then the model with the higher accuracy should be chosen. However, if the task requires a balance between precision and recall, then the F1-score, which is the harmonic mean of precision and recall, should be used. The model with the higher F1-score should be chosen in this case.

It's worth noting that the dataset and the problem you're trying to solve might lead to one algorithm being better suited than the other. It's always a good idea to try different algorithms and compare their performance.

So, it is solely dependent on the user's requirements. 

To know more about the parameters, you can refer to the Medium's blog [here](https://towardsdatascience.com/comparing-random-forest-and-gradient-boosting-d7236b429c15)

### CONGRATULATIONS!!! YOU HAVE REACHED THIS FAR, BUT

<img src="https://user-images.githubusercontent.com/110394695/212535375-aec902ea-412e-4275-b097-98876a7a8aa8.gif" width="1000" height="400">


**Ping me on my socials below, Let's connect!!!**


<a href="https://www.linkedin.com/in/kshitiz-pandya-687659230/">
  <img alt="LinkedIn" src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" />
</a>
<a href="https://github.com/KshitizPandya">
  <img alt="Github" src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" />
</a>
<a href="https://kshitiz.pandya@gmail.com/">
  <img alt="Gmail" src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" />
</a>
<a href="https://instagram.com/kshitiz._.pandya">
  <img alt="Instagram" src="https://img.shields.io/badge/Instagram-%23E4405F.svg?style=for-the-badge&logo=Instagram&logoColor=white" />
</a>
</a>
<a href="https://discord.gg/user/KENUBEE#1045">
  <img alt="Discord" src="https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white" />
</a>

