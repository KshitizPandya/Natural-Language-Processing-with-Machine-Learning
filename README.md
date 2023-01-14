# Natural Language Processing with Machine Learning


This repository covers the basic understanding of what is **Natural Language Processing (NLP)** and how we can achieve it using **Natural Language Tool Kit (NLTK**).

We have used dataset given named as `SMSSpamCollection.tsv` which is a tab-seperated-values file. Our dataset is a semistructured text data of various SMSs which are marked as ham or spam. 


Note - While reading this documentation, you can directly click on the topic hyperlinked to 
its respective `.ipynb` files.

[Click Here](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning), to directly refer to the repository.

[Click Here](https://drive.google.com/file/d/1QIDyNzIfTYHR92-sQlRys_gpycBKnkLG/view?usp=sharing), to download the dataset directly.

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

![output_3](https://user-images.githubusercontent.com/110394695/211855348-2c59d74a-0667-4e07-9c04-7f58168301e5.png)


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



### [Feature Transformationn](https://github.com/KshitizPandya/Natural-Language-Processing-with-Machine-Learning/blob/main/4.%20Feature%20Engineering/4.2.%20Transformation.ipynb)
 


