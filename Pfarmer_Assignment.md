---
geometry: margin=30mm
title: "Machine Learning Final Assignment"
author: "Student Name: Patrick Farmer       Student Number: 20331828"
date: "Date: 30-11-2024"
---

# Part 1

### Introduction

This Assignment aims to generate nursery rhyme rhythms given a simplified input dataset. The dataset has characters that represent different notes. Each note will be played for 0.5 seconds, if the same character appears a couple of times in a row it will be played for longer. The project is built upon the GPT used in lab9 with some addition and changes made to improve the model's performance for this task. When the initial model was used the output already sounded perfectly feasible to my ear, for that reason during this lab measurable metrics will be used to evaluate the model performance. Due to the prediction being quite simple a number of different approaches were taken and the results across each will be compared.\
The first of which approaches was to simply adjust the hyperparameters to fit the dataset better. The second approach was to first train the model on the pitch only and remove rhythm, then after that fine tune the model on the full dataset which included rhythm. The third and final approach that was used and produced acceptable results was to train a model on the pitch and rhythm separately and then combine the two models to generate the final output.\

### Data Preprocessing and feature engineering

There was a couple forms of data augmentation applied to the dataset overall for every method. The first of which was a simple noise generator which would for two in every 10 characters replace the character with a random character from the dataset. The second form of data augmentation was to stretch and shrink the patterns in the dataset. This works by first finding the common patterns in the dataset and when one occurs it will be stretched or shrunk by a random amount within some threshold. This was done to emphasise the patterns (rhythms) in the dataset.\
For some of the methods additional pre-requisite steps were required. For the method to train the model on pitch only first the dataset was preprocessed to remove all timing data. This was done by simply collapsing all sequential occurances of the same character into one. The model was then first trained on this dataset and was trained secondly on the full dataset.\
For the method to train the model on pitch and rhythm separately the dataset was split into two datasets, the first which contained only pitch was processed as mentioned above and the dataset was used again here. The second dataset that contained only rhythm was preprocesses in the same script. It would replace the characters with numbers representing the length of the note. Models were then trained on both of these datasets. The separate models were very accurate for their tasks as the task was simpler than the combined task. They were then combined together to generate the final output by rebuilding the data in the same way it was split. This is effectively a multiplication of the datasets. i.e.:

```
timing = "1113111
pitch = "CDEFGAB"
output = "CDEFFFGAB"
```


### Machine Learning methodology

### Evaluation

# Part 2

### What is an ROC curve? How can it be used to evaluate the performance of a classifier compared with a baseline classifier? Why would you use an ROC curve instead of a classification accuracy metric?

### Give two examples of situations where a linear regression would give inaccurate predictions. Explain your reasoning and what possible solutions you would adopt in each situation.

### The term 'kernel' has different meanings in SVM and CNN models. Explain the two different meanings. Discuss why and when the use of SVM kernels and CNN kernels is useful, as well as mentioning different types of kernels.

### In k-fold cross-validation, a dataset is resampled multiple times. What is the idea behind this resampling i.e. why does resampling allow us to evaluate the generalisation performance of a machine learning mode. Give a small example to illustrate. Discuss when it is and it is not appropriate to use k-fold cross-validation.

Sequential pitch, timing, combined and resultant vocubulary adjustment
Cosine similarity loss
Added temperature for creativity
Reduced block size from text generation as context is more localised
Larger batch sizes to encourage faster convergence
Convergence mux faster so heavily decreased iterations and evaluation intervals
Reduced dropout as rhytms are less complex
Sounded too similar to input so increased temperature and added noise to improve generalisation
Added pattern stretch and shrink to improve generalisation and torch cat all datasets together
Added 1D convolution layers before Transformer blocks to extract local rhythm patterns
Evaluation
Separation perhaps more useful if more complex rhythms
