---
geometry: margin=20mm
title: "Machine Learning Final Assignment"
author: "Student Name: Patrick Farmer       Student Number: 20331828"
date: "Date: 30-11-2024"
---


![](https://www.tcd.ie/media/tcd/site-assets/images/tcd-logo.png)

\clearpage

# Part 1

## Introduction

This Assignment aims to generate nursery rhyme rhythms given a simplified input dataset. The dataset has characters that represent different notes. Each note will be played for 0.5 seconds, if the same character appears a couple of times in a row it will be played for longer. The project is built upon the GPT used in lab9 with some addition and changes made to improve the model's performance for this task. When the initial model was used the output already sounded perfectly feasible to my ear, for that reason during this lab measurable metrics will be used to evaluate the model performance. Due to the prediction being quite simple a number of different approaches were taken and the results across each will be compared.\
The first of which approaches was to simply adjust the hyperparameters to fit the dataset better. The second approach was to first train the model on the pitch only and remove rhythm, then after that fine tune the model on the full dataset which included rhythm. The third and final approach that was used and produced acceptable results was to train a model on the pitch and rhythm separately and then combine the two models to generate the final output.\

## Data Preprocessing and feature engineering

There was a couple forms of data augmentation applied to the dataset overall for every method. The first of which was a simple noise generator which would for two in every 10 characters replace the character with a random character from the dataset. The second form of data augmentation was to stretch and shrink the patterns in the dataset. This works by first finding the common patterns in the dataset and when one occurs it will be stretched or shrunk by a random amount within some threshold. This was done to emphasise the patterns (rhythms) in the dataset.\
For some of the methods additional pre-requisite steps were required. For the method to train the model on pitch only first the dataset was preprocessed to remove all timing data. This was done by simply collapsing all sequential occurances of the same character into one. The model was then first trained on this dataset and was trained secondly on the full dataset.\
For the method to train the model on pitch and rhythm separately the dataset was split into two datasets, the first which contained only pitch was processed as mentioned above and the dataset was used again here. The second dataset that contained only rhythm was preprocesses in the same script. It would replace the characters with numbers representing the length of the note. Models were then trained on both of these datasets. The separate models were very accurate for their tasks as the task was simpler than the combined task. They were then combined together to generate the final output by rebuilding the data in the same way it was split. This is effectively a multiplication of the datasets. i.e.:

```
timing = "1113111
pitch = "CDEFGAB"
output = "CDEFFFGAB"
```

## Machine Learning methodology

The first change that was made was a change to the loss function. A combination of two loss functions was used. A weighted sum of the cosine embedding loss and the cross entropy loss. The cosine embedding loss minimises the difference between the sequence generated and the target sequence which captures the rhythm as a whole. The cross entropy loss is more specifically for the timing to capture the beat fo the rhythm. The weighting is weighted 70% cosine loss and 30% cross entropy loss.

Temperature was also added to increase the creativity of the model. Temperature adjusts the smoothness of the probability distribution. A higher temperature will make the model more creative but less accurate. It is implemented by dividing the logits by the temperature before applying the softmax function. To improve the accuracy the temperature was reduced to 0.6 but if the the task for the model required more diversity a higher temperature could be used.

The Hyperparameters were adjusted to fit the dataset better. The model as a whole was made much more simple as the task is a lot repetitive and structured than the previous task the gpt.py was used for. Three different sets of hyperparameters were used to compare against each other.

The **first set of hyperparameters** is a very light weight set of parameters, the model uses only 490 parameters. The hyperparameters are as follows:

```python
batch_size = 64         # The batch size was kept large as it keeps the convergence stable
block_size = 4          # The block size was reduced to 4 as the patterns are very localised.
# This would not capture the larger patterns and this will be compared with the other models
max_iters = 20          # The max iterations was reduced to 20 as the model converged very quickly
eval_interval = 2       # The evaluation interval was reduced to 2 as the model converged quickly
learning_rate = 1e-4    # The learning rate was kept low as the model is simple
eval_iters = 1          # The evaluation iterations was reduced to 1 as the model converged quickly
n_embd = 8              # The number of embeddings was kept very low as there was not overly 
# complex patterns in the dataset
n_head = 1              # The number of heads was kept low as there was not overly complex 
# relationships between tokens in the dataset
n_layer = 1             # The number of layers was kept small as the dataset was simple and could
# easily be overfit
dropout = 0.1           # The dropout was set low as there was a large amount of data and the model
# was simple so it is not likely to overfit
```

The **second set of hyperparameters** was slightly more complex and allowed for the same heavy localisation focus but allowed more complex patterns and relationships to be captured. The model still had a reasonably small parameter count at 20,000. The model was also run for a few more iterations. The hyperparameters are as follows:

```python
batch_size = 64         # The batch size was kept large as it keeps the convergence stable    
block_size = 4          # The block size was kept at 4 to mirror the local focus in the first model    
max_iters = 100         # The max iterations was increased to 100 as the model was more complex    
eval_interval = 10      # The evaluation interval was increased to 10 as the model was more complex        
learning_rate = 1e-4    # The learning rate was kept low as the model is still relatively simple            
eval_iters = 10         # The evaluation iterations was increased to 10 as the model was more complex    
n_embd = 32             # The number of embeddings was increased to 32 to capture more complex patterns
n_head = 4              # The number of heads was increased to 4 to capture more complex relationships
# between tokens
n_layer = 1             # The number of layers was kept small as the dataset was simple and could
# easily be overfit
dropout = 0.1           # The dropout was set low as there was a large amount of data and the model was
# simple so it is not likely to overfit    
```

The **third set of hyperparameters** was the most complex, it tested how the model would perform if it was allowed to capture bigger patterns in addition to more complex patterns. This model has a parameter count of 40,000. The hyperparameters are as follows:

```python
batch_size = 64         # The batch size was kept large as it keeps the convergence stable      
block_size = 32         # The block size was increased to 32 to capture larger patterns    
max_iters = 400         # The max iterations was increased to 400 as the model was more complex    
eval_interval = 40      # The evaluation interval was increased to 40 as the model was more complex        
learning_rate = 1e-4    # The learning rate was kept low as the model is still relatively simple            
eval_iters = 40         # The evaluation iterations was increased to 40 as the model was more complex    
n_embd = 32             # The number of embeddings was increased to 32 to capture more complex patterns
n_head = 4              # The number of heads was increased to 4 to capture more complex relationships
# between tokens
n_layer = 2             # The number of layers was increased to 2 to capture more complex patterns
dropout = 0.1           # The dropout was set low as there was a large amount of data and the model was
# simple so it is not likely to overfit    
```

In order to extract the local rhythm patterns more effectively 1D convolutional layers were added before the transformer blocks. This works because the convolutional layers can extract the local patterns as they would in an image classifier and the transformer blocks can then more easily capture the relationships between the local patterns.

## Evaluation

It was mentioned in the introduction that it would not be possible to make a subjective evaluation of the model as all of the outputs sounded feasible to the ear. For that reason the models were compared with a number of metrics. The metrics that were used were a few standard machine learning metrics F1-score, precision and recall. Also a few more less conventional metrics, character distribution distance, pattern length distribution distance and pattern distribution distance.

The output was broken into patterns of length 2-4. Every pattern in both the target and the output was logged into a dictionary. Each of the patterns in the output was checked against the model patterns and if it was there it was incremented. This gives us our true positives. The true positives and false negatives are then just the length of the model pattern array. Using these two values the pattern recall can be calculated. The same process was followed for precision where the true positives and false positives are the length of the output pattern array. The F1-score is then calculated using the precision and recall.

The character distribution metric was calculated by taking the distance between the character distribution of the target and the output. This was a sanity check metric to make sure that the model wasn't just asking as a dummy classifier. The pattern length distribution metric was calculated by taking the distance between the distribution of the length of sequential notes in the target and the output. This was a metric to make sure that the model was capturing the rhythm. The pattern distribution metric was calculated by taking the distance between the distribution of the patterns in the target and the output. This was a metric to make sure that the model was capturing the patterns in the dataset.

A baseline model was also created to compare the models against. The baseline model was a simple model that would predict the most common character with the most common sequence length, which happened to be 'f'.

Taking first the recall comparison. We get the below chart:(higher is better)\
![Figure 1](<images/Pattern Recall.png>)\
Most of the models are close to each other but the best 2 results are both using the first set of hyperparameters with the better of the two being the model that was trained on pitch only first. This is because the model was able to capture the patterns in the dataset better. We do see however that some of the models fell apart to give a very low recall. In the case of the fine-tune parameter 3 that is because the model is just predicted the 1 character for the entire output very similarly to the baseline model. We also see that the simple model 1 for the separate model approach performed very poorly. This is because the model was able to capture the pitch with a simple model but not catch the timing, the result of this is it predicted very long notes which was very off rhythm.

For the precision comparison we get the below chart:(higher is better)\
![Figure 2](<images/Pattern Precision.png>)\
The precision tells use very little as almost all of the models have a precision between 0.9 and 1. That is because the model has a low temperature and is unlikely to produce false positives and would be more likely to produce false negatives. We only see here the fine-tune model 3 has a precision of 1 due to the model predicting just the one character.

For the F1-score comparison we get the below chart:(higher is better)\
![Figure 3](<images/Pattern F1 Score.png>)\
The F1-score is a combination of the precision and recall. We just saw above that the precision is almost entirely the same which is why our F1-score graph very closely reflects the recall graph. Due to the unaugmented model having a slightly better precision than the fine-tune models they close the gap slightly in the F1-score but the fine-tune model approach still has the best F1-score.

For the character distribution distance comparison we get the below chart:(lower is better)\
![Figure 4](<images/Character Distribution Distance.png>)\
The character distribution distance is mainly a sanity check that shows if a model is completely off mark and we can see this with the fine-tune model 3 and the baseline model. The baseline model and fine-tune model 3 having the same character distribution distance is because they are both predicting the same character for the entire output. We can see that the separate model 1 has a normal distance and this is because it was able to capture the pitch but not the timing.

For the pattern length distribution distance comparison we get the below chart:(lower is better)\
![Figure 5](<images/Pattern Length Distribution Distance.png>)\
The pattern length distribution distance is a metric to show if the model is capturing the timing correctly. As predicted above the separate model 1 has a very high distance as it was unable to capture the timing correctly. We can also see that the baseline model is not massively off. This is due to the model predicting the most common sequence length which was 1. This is an example of why we need to use multiple metrics to evaluate a model.

For the pattern distribution distance comparison we get the below chart:(lower is better)\
![Figure 6](<images/Pattern Distribution Distance.png>)\
The pattern distribution distance is a metric to show if the model is capturing the patterns in the dataset. We can see that the same models shown to be performing well in the f1-score metric are also performing well here. This is because both metrics are comparing similar things but the pattern distribution distance also takes into account the count of the patterns occurring in the dataset. Due to this data being very repetitive most models are generating the patterns at the correct frequency so the metric gives similar results to the f1-score for this reason. It could for another dataset provide a more useful insight though.

## Discussion

The results show that the fine-tune model which was trained first on pitch and then on both pitch and timing together performed the best, this was due to it getting the pitch correct in a simplified setting and using the obtained knowledge to finish the training. The poorest model was the model for which pitch and timing were trained separately, reading through some of the output this is because although the model produced a good model of timing and produced a good model of pitch it did not know how to match the pitch to the timing correctly, instead a naive multiplication was performed. This approach could have been much improved by training another model to combine them correctly. The unaugmented data model also performed well, however with a more complex dataset it would likely have fallen off worse than the other two model types. 

\clearpage

# Part 2

### What is an ROC curve? How can it be used to evaluate the performance of a classifier compared with a baseline classifier? Why would you use an ROC curve instead of a classification accuracy metric?

An ROC curve plots the True Positive Rate against the False Positive Rate at different thresholds (where the threshold is the ratio of Positives to Negatives). The True Positive Rate is the True Positives over the True Positives and False Negatives. The False Positive Rate is the False Positives over the False Positives and True Negatives. The ROC curve is useful for comparing the model against a baseline classifier which appears on the graph as a diagonal line, where the area between the ROC curve and the baseline is a measure of how much better than the baseline model the ROC curve is performing. The ROC curve gives you more information than a classification accuracy metric as it shows how the model is performing at different thresholds. This is useful as the classification accuracy metric only shows how the model is performing at one threshold.

### Give two examples of situations where a linear regression would give inaccurate predictions. Explain your reasoning and what possible solutions you would adopt in each situation.

Linear Regression is as the name suggests suited for linear relationships. When there is a non-linear relationship between variables linear regression will give inaccurate predictions. For example for some data with a quadratic relationship the linear regression will separate the data into two groups by drawing a line through the middle of the data that does not in any way capture the relationship between the variables.\
Linear Regression is also not suited for data with a high amount of noise, specifically outliers as the model will try to fit the line to the outliers. This can be solved with pre-requisite data cleaning but without additional help Linear Regression will perform poorly on this data.

### The term 'kernel' has different meanings in SVM and CNN models. Explain the two different meanings. Discuss why and when the use of SVM kernels and CNN kernels is useful, as well as mentioning different types of kernels.

In an SVM (Support Vector Machine) a kernel refers to a function that transforms the data into a higher dimensional space that allows enables the SVM to find a hyperplane that does a better job at separating the data. This is useful as it allows the SVM to find more complex patterns in the data. A couple examples of kernels are the RBF kernel and the sigmoid kernel.\
In a CNN (Convolutional Neural Network) a kernel is a matrix that is used to perform convolution on the input data. The kernel may be used to find edges, smooth an image or in larger networks to find more complex patterns such as faces. A couple examples of kernels are the Sobel kernel (for finding edges) and the Gaussian kernel (for smoothing an image).

### In k-fold cross-validation, a dataset is resampled multiple times. What is the idea behind this resampling i.e. why does resampling allow us to evaluate the generalisation performance of a machine learning mode. Give a small example to illustrate. Discuss when it is and it is not appropriate to use k-fold cross-validation.

K-fold cross-validation will divide the dataset into k subsets. It will then train the model on all but one of the subsets and then test this trained model onto the remaining data set. This can be repeated k times. This will gives a better evaluation of the generalisation performance of the model as it will be trained and tested on multiple different datasets.\
For example take a dataset with 1000 samples. We can split this into 5 so each dataset has 200 samples. We can then use dataset 1-4 as the training dataset of 800 samples and dataset 5 as the testing dataset of 200 samples. We can then repeat this 5 times so that each dataset is used as the testing dataset once.\
K-fold cross-validation is well suited to small datasets as it allows the developer to get a better idea of how well the model generalises even with a small dataset. It would still provide the same information on a larger dataset but it would be less valuable information and much more computationally expensive so it would not be appropriate for a large dataset. It would also not be appropriate for a dataset that relies on the order of the data as when using k-fold cross-validation the data is shuffled.

# Appendix

## gpt.py

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter
import sys
import argparse
import os

parser = argparse.ArgumentParser(description='GPT Language Model')
parser.add_argument('--path', type=str, help='Path to the input file')
parser.add_argument('--parameters', type=int, help='Hyperparameters set')
parser.add_argument('--model-path', type=str, help='Path to the model file')
parser.add_argument('--no-train', type=str, help='Skip training')
parser.add_argument('--log-path', type=str, help='Path to the log file')
parser.add_argument('--melody-path', type=str, help='Path to the melody file')
parser.add_argument('--data-type', type=str, help='Type of data')
args = parser.parse_args()

if args.no_train != "True":
    log_file = args.log_path
    with open(log_file, 'w') as f:
        f.write('')

# hyperparameters
# Original set of hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Hyperparameters set 1
if args.parameters == 6:
    batch_size = 128
    block_size = 16
    max_iters = 500
    eval_interval = 50
    learning_rate = 2e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 20
    n_embd = 128
    n_head = 4 
    n_layer = 4 
    dropout = 0.05
# ------------

# Hyperparameters set 2
if args.parameters == 5:
    batch_size = 128
    block_size = 64
    max_iters = 500
    eval_interval = 50
    learning_rate = 2e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 20
    n_embd = 128
    n_head = 3
    n_layer = 3
    dropout = 0.05

# Hyperparameters set 3
if args.parameters == 4:
    batch_size = 128
    block_size = 32
    max_iters = 500
    eval_interval = 50
    learning_rate = 2e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 50
    n_embd = 172
    n_head = 2
    n_layer = 2
    dropout = 0.05
    
# Hyperparameters set 4
if args.parameters == 1:
    batch_size = 64
    block_size = 4
    max_iters = 20
    eval_interval = 2
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 1
    n_embd = 8
    n_head = 1
    n_layer = 1
    dropout = 0.1
    
# Hyperparameters set 5
if args.parameters == 2:
    batch_size = 64
    block_size = 4
    max_iters = 100
    eval_interval = 10
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 10
    n_embd = 32
    n_head = 4
    n_layer = 1
    dropout = 0.1
    
# Hyperparameters set 6
if args.parameters == 3:
    batch_size = 64
    block_size = 32
    max_iters = 400
    eval_interval = 40
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 40
    n_embd = 32
    n_head = 4
    n_layer = 1
    dropout = 0.1
    
if args.data_type == "fine_tune":
    learning_rate *= 2

def add_noise(data, chars):
    augmented = []
    for char in data:
        if torch.rand(1).item() < 0.2:
            continue 
        augmented.append(char)
        if torch.rand(1).item() < 0.2:
            augmented.append(stoi[chars[torch.randint(len(chars), (1,)).item()]])
    return torch.tensor(augmented, dtype=torch.long)

def pattern_stretch_shrink(data):
    stretched = []
    patterns = {}
    pattern_length = 2
    for i in range(len(data)):
        if i != len(data) - 1:
            pattern = data[i:i + pattern_length - 1]
            if pattern in patterns:
                patterns[pattern] += 1
            else:
                patterns[pattern] = 1
    
    pattern_sum = sum(patterns.values())        
    pattern_avg = pattern_sum / len(patterns)
    patterns_to_remove = []
    for pattern in patterns:
        if patterns[pattern] < pattern_avg:
            patterns_to_remove.append(pattern)
    for pattern in patterns_to_remove:
        patterns.pop(pattern)
        
    for i in range(len(data)):
        stretched.append(data[i])
        if i != len(data) - 1:
            pattern = data[i:i + pattern_length - 1]
            if pattern in patterns:
                stretched.append(data[i + 1])
                stretched.append(data[i])
            
    return stretched
    
    
        
        
        
    
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

input_file = args.path

with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

text = pattern_stretch_shrink(text)

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
data = torch.cat((data, add_noise(data, chars)))
# data = torch.cat((data, add_noise(data, chars)))

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=n_embd, out_channels=n_embd, kernel_size=5, padding=2)
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            if args.data_type != 'timing':
                loss_a = F.cosine_embedding_loss(logits, F.one_hot(targets, num_classes=vocab_size).float(), torch.ones(logits.size(0), device=device))
                loss_b = F.cross_entropy(logits, targets)
                loss = loss_a * 0.7 + loss_b * 0.3
            else:
                loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # Add temperature to the logits
            logits /= 0.3
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx_next = torch.clamp(idx_next, 0, vocab_size - 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


if args.no_train == None:    
    model = GPTLanguageModel()
    
    
    if os.path.exists(args.model_path) and args.data_type == "fine_tune":
        print(f"Loading model from {args.model_path}...")
        # Load state_dict
        checkpoint = torch.load(args.model_path)
        
        # Adjust model to fit checkpoint vocabulary size
        checkpoint_vocab_size = checkpoint['token_embedding_table.weight'].size(0)
        model_vocab_size = model.token_embedding_table.weight.size(0)
        
        if checkpoint_vocab_size != model_vocab_size:
            print(f"Adjusting model vocabulary size from {model_vocab_size} to {checkpoint_vocab_size}")
            # Update embedding and output layers
            model.token_embedding_table = nn.Embedding(checkpoint_vocab_size, model.token_embedding_table.weight.size(1))
            model.lm_head = nn.Linear(model.lm_head.weight.size(1), checkpoint_vocab_size)

        # Load updated state_dict
        model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    else:
        print("Training model...")
    m = model.to(device)
    
    # print the number of parameters in the model
    with open(log_file, 'a') as f:
        f.write(f"{sum(p.numel() for p in m.parameters())/1e6} M parameters\n")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Write the vocabulary size to the log
    with open(log_file, 'a') as f:
        f.write(f"Vocabulary size: {vocab_size}\n")
        
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()

            # Append losses during training
            with open(log_file, 'a') as f:
                f.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    if args.melody_path != None:
        generated_text = decode(m.generate(context, max_new_tokens=5000)[0].tolist())
        # print(generated_text)
        with open(args.melody_path, 'w') as f:
            f.write(generated_text)

    # Save the model
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")
    
    most_frequent_char = max(set(text), key=text.count)
    most_frequent_char_idx = stoi[most_frequent_char]
    
    melody_path = args.melody_path
    melody_leaf = melody_path.split('/')[-1]
    melody_leaf = 'baseline_' + melody_leaf
    baseline_path = os.path.join('/'.join(melody_path.split('/')[:-1]), melody_leaf)

    with open(baseline_path, 'w') as f:
        baseline_context = torch.zeros((1, 1), dtype=torch.long, device=device).fill_(most_frequent_char_idx)
        baseline_generated_text = decode([baseline_context.squeeze().item()])
        f.write(baseline_generated_text)



if args.no_train == "True":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = GPTLanguageModel().to(device)

    # Load the checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)

    # Retrieve embedding and output weights from checkpoint
    embedding_weight = checkpoint['token_embedding_table.weight']
    output_weight = checkpoint['lm_head.weight']
    output_bias = checkpoint['lm_head.bias']

    # Check for vocabulary size mismatch
    checkpoint_vocab_size = embedding_weight.shape[0]
    if checkpoint_vocab_size != vocab_size:
        print("Adjusting model for different vocab size...")

        # Adjust token embedding layer
        new_embedding_weight = torch.zeros((vocab_size, n_embd), device=device)
        new_embedding_weight[:min(vocab_size, checkpoint_vocab_size), :] = embedding_weight[:min(vocab_size, checkpoint_vocab_size), :]
        model.token_embedding_table = nn.Embedding(vocab_size, n_embd).to(device)
        model.token_embedding_table.weight = Parameter(new_embedding_weight)

        # Adjust lm_head output layer weights
        new_output_weight = torch.zeros((vocab_size, n_embd), device=device)
        new_output_weight[:min(vocab_size, checkpoint_vocab_size), :] = output_weight[:min(vocab_size, checkpoint_vocab_size), :]
        model.lm_head = nn.Linear(n_embd, vocab_size).to(device)
        model.lm_head.weight = Parameter(new_output_weight)

        # Adjust lm_head output layer biases
        new_output_bias = torch.zeros(vocab_size, device=device)
        new_output_bias[:min(vocab_size, checkpoint_vocab_size)] = output_bias[:min(vocab_size, checkpoint_vocab_size)]
        model.lm_head.bias = Parameter(new_output_bias)
    else:
        # No mismatch, load state dictionary directly
        model.load_state_dict(checkpoint)

    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {args.model_path}")

    # Prepare the validation dataset
    with open(args.path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Process the text data
    data = torch.tensor(encode(text), dtype=torch.long)
    val_data = data

    loss = estimate_loss()
    val_loss = loss['val'].item()
  
    results_log = args.log_path
    with open(results_log, 'w') as f:
        f.write('Vocabulary size: {}\n'.format(vocab_size))
        f.write("Trained Model:\n")
        f.write(f"Validation Loss: {val_loss:.4f}\n")

    val_data = val_data.to(device)

    # Identify the most frequent character and its index
    most_frequent_char = max(set(text), key=text.count)
    most_frequent_char_idx = stoi[most_frequent_char]

    # Generate dummy predictions (all predictions set to the index of the most frequent character)
    dummy_predictions = torch.full((len(val_data),), most_frequent_char_idx, dtype=torch.float, device=device)

    # Calculate dummy loss
    # Ensure all tensors (both predictions and targets) are on the same device and of compatible types
    dummy_loss = F.cross_entropy(dummy_predictions.unsqueeze(0), val_data.unsqueeze(0).float()).item()

    # Log the dummy classifier loss
    with open(results_log, 'a') as f:
        f.write("Baseline Dummy Classifier:\n")
        f.write(f"Validation Loss: {dummy_loss:.4f}\n")
```

## output_comparison.py

```python
import argparse
import csv
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Compare output files and generate a CSV report.')
parser.add_argument('--truth', required=True, help='Path to the truth file')
parser.add_argument('--outputs', required=True, nargs='+', help='Paths to the output files')
parser.add_argument('--output-csv', required=True, help='Path to the output CSV file')
args = parser.parse_args()

csv_headers = ['File', 'Character Distribution Distance', 'Pattern Distribution Distance', 'Pattern Length Distribution Distance', 'Pattern Recall', 'Pattern Precision', 'Pattern F1 Score']
with open(args.output_csv, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(csv_headers)

def get_char_distribution(data):
    char_distribution = {}
    for char in data:
        if char in char_distribution:
            char_distribution[char] += 1
        else:
            char_distribution[char] = 1
    total_chars = sum(char_distribution.values())
    for char in char_distribution:
        char_distribution[char] /= total_chars
    return char_distribution

def get_patterns(data):
    patterns = {}
    for i in range(len(data)):
        for j in range(2, 4):
            if i + j <= len(data):
                pattern = data[i:i + j]
                if pattern in patterns:
                    patterns[pattern] += 1
                else:
                    patterns[pattern] = 1
    total_patterns = sum(patterns.values())
    for pattern in patterns:
        patterns[pattern] /= total_patterns
    return patterns 

def get_sequence_lengths(data):
    lengths = {}
    i = 0
    length = 1
    last_char = None
    for char in data:
        if last_char:
            if char == last_char:
                length += 1
            else:
                if length in lengths:
                    lengths[length] += 1
                else:
                    lengths[length] = 1
                length = 1
        last_char = char
    if length in lengths:
        lengths[length] += 1
    else:
        lengths[length] = 1
    total_lengths = sum(lengths.values())
    for length in lengths:
        lengths[length] /= total_lengths
    return lengths

truth_data = ''
with open(args.truth, 'r') as file:
    truth_data = file.read()
    truth_char_distribution = get_char_distribution(truth_data)
    truth_patterns = get_patterns(truth_data)
    truth_sequence_lengths = get_sequence_lengths(truth_data)
    
    results = {}

    for output_file in args.outputs:
        with open(output_file, 'r') as file:
            model_data = file.read()
            model_char_distribution = get_char_distribution(model_data)
            char_distance = 0
            for char in truth_char_distribution:
                truth_freq = truth_char_distribution.get(char, 0)
                model_freq = model_char_distribution.get(char, 0)
                char_distance += abs(truth_freq - model_freq)
            char_distance /= len(truth_char_distribution)
                
            pattern_distance = 0
            model_patterns = get_patterns(model_data)
            for pattern in truth_patterns:
                truth_freq = truth_patterns.get(pattern, 0)
                model_freq = model_patterns.get(pattern, 0)
                pattern_distance += abs(truth_freq - model_freq)
            pattern_distance /= len(truth_patterns)
                
            lengths_distance = 0
            model_sequence_lengths = get_sequence_lengths(model_data)
            for length in truth_sequence_lengths:
                truth_freq = truth_sequence_lengths.get(length, 0)
                model_freq = model_sequence_lengths.get(length, 0)
                lengths_distance += abs(truth_freq - model_freq)
            lengths_distance /= len(truth_sequence_lengths)
                
            pattern_count = 0
            for pattern in truth_patterns:
                if pattern in model_patterns:
                    pattern_count += 1
            if len(model_patterns) == 0:
                pattern_recall = 0
                pattern_precision = 0
                pattern_f1_score = 0
            else:
                pattern_recall = pattern_count / len(truth_patterns)
                pattern_precision = pattern_count / len(model_patterns)
                pattern_f1_score = 2 * (pattern_precision * pattern_recall) / (pattern_precision + pattern_recall)
                    
            output_file_name = output_file.split('/')[-1]
            results[output_file_name] = {
                'Character Distribution Distance': char_distance,
                'Pattern Distribution Distance': pattern_distance,
                'Pattern Length Distribution Distance': lengths_distance,
                'Pattern Recall': pattern_recall,
                'Pattern Precision': pattern_precision,
                'Pattern F1 Score': pattern_f1_score
            }

        with open(args.output_csv, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([output_file_name, f"{char_distance:.4f}", f"{pattern_distance:.4f}", f"{lengths_distance:.4f}", f"{pattern_recall:.4f}", f"{pattern_precision:.4f}", f"{pattern_f1_score:.4f}"])

for metric in csv_headers[1:]:
    plt.figure(figsize=(10, 5))
    values = []
    for key in results:
        values.append(results[key][metric])
    
    non_zero_keys = []
    non_zero_values = []
    for key, value in zip(results.keys(), values):
        if value != 0:
            non_zero_keys.append(key)
            non_zero_values.append(value)
    
    plt.bar(non_zero_keys, non_zero_values)
    plt.title(metric)
    plt.xlabel('Output Files')
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'images/{metric}.png')
```

## combine.py

```python
import argparse

parser = argparse.ArgumentParser(description='Combine pitch and timing files into an output file.')
parser.add_argument('--pitch', required=True, help='Path to the pitch file')
parser.add_argument('--timing', required=True, help='Path to the timing file')
parser.add_argument('--output', required=True, help='Path to the output file')

args = parser.parse_args()

with open(args.pitch, 'r') as pitch_file:
    pitch_data = pitch_file.read()
    
with open(args.timing, 'r') as timing_file:
    timing_data = timing_file.read()
    
output_data = ''
    
while True:
    output_data += (pitch_data[0] * int(timing_data[0]))
    pitch_data = pitch_data[1:]
    timing_data = timing_data[1:]
    if not pitch_data or not timing_data:
        break

with open(args.output, 'w') as output_file:
    output_file.write(output_data)
```

## data_preprocess.py

```python
import argparse


parser = argparse.ArgumentParser(description='GPT Language Model')
parser.add_argument('--path', type=str, help='Path to the input file')
parser.add_argument('--pitch-path', type=str, help='Path to the pitch file')
parser.add_argument('--timing-path', type=str, help='Path to the timing file')
parser.add_argument('--interleaved-path', type=str, help='Path to the interleaved file')
parser.add_argument('--unaltered-path', type=str, help='Path to the unaltered file')
args = parser.parse_args()

with open(args.path, 'r') as file:
    input_text = file.read()
    
pitch_data = ''
timing_data = ''
interleaved_data = ''
count = 1
last_char = None
for char in input_text:
    if last_char:
        if char == last_char:
            pitch_data += ''
            count += 1
        else:
            pitch_data += char
            timing_data += str(count)
            interleaved_data += 'T' + str(count) + 'C' + char
            count = 1
    last_char = char


with open(args.pitch_path, 'w') as pitch_file:
    pitch_file.write(pitch_data)
    
    
with open(args.timing_path, 'w') as timing_file:
    timing_file.write(timing_data)
    
with open(args.unaltered_path, 'w') as unaltered_file:
    unaltered_file.write(input_text)
    
with open(args.interleaved_path, 'w') as interleaved_file:
    interleaved_file.write(interleaved_data)
```

## run

```bash
#!/bin/bash
basepath=inputs/finalAssignment_musicDataset

mkdir -p temp

range=3

python scripts/data_preprocess.py --path $basepath/inputMelodiesAugmented.txt --pitch-path temp/pitch.txt --timing-path temp/timing.txt --interleaved-path temp/interleaved.txt --unaltered-path temp/unaltered.txt

for i in $(seq 1 $range); do
    modelPath=Models/model_melodies_$i
    rm -rf $modelPath
    python gpt.py --path temp/pitch.txt --parameters $i --model-path $modelPath --log-path Logs/result_melodies_pitch_$i.log --melody-path $basepath/melody_pitch_$i.txt --data-type pitch
    python gpt.py --path temp/unaltered.txt --parameters $i --model-path $modelPath --log-path Logs/result_melodies_fine_tune_$i.log --melody-path $basepath/melody_fine_tune_$i.txt --data-type fine_tune
    python gpt.py --path temp/timing.txt --parameters $i --model-path $modelPath --log-path Logs/result_melodies_timing_$i.log --melody-path $basepath/melody_timing_$i.txt --data-type timing
    

    python scripts/combine.py --pitch $basepath/melody_pitch_$i.txt --timing $basepath/melody_timing_$i.txt --output $basepath/melody_multiplied_$i.txt

    #python gpt.py --path temp/interleaved.txt --parameters $i --model-path $modelPath --log-path Logs/result_melodies_interleaved_$i.log --melody-path $basepath/melody_interleaved_$i.txt --data-type both

    #python scripts/postprocess.py --path $basepath/melody_interleaved_$i.txt
done

for i in $(seq 1 $range); do
    modelPath=Models/model_melodies_no_aug_$i
    rm -rf $modelPath
    python gpt.py --path temp/unaltered.txt --parameters $i --model-path $modelPath --log-path Logs/result_melodies_no_aug_$i.log --melody-path $basepath/melody_no_aug_$i.txt --data-type both
done

# LowestLost=100000000
# bestIndex=-1
# for i in $(seq 1 $range); do
#     loss=$(tail -n 1 Logs/result_melodies_$i.log | awk '{print $NF}')
#     if (( $(awk "BEGIN {print ($loss < $LowestLost)}") )); then
#         LowestLost=$loss
#         bestIndex=$i
#         BestModel=Models/model_melodies_$i
#     fi
# done
#cp $basepath/melody_$bestIndex.txt $basepath/melody_best.txt

melodies="melody_no_aug_ melody_multiplied_ melody_fine_tune_" 
output_files=""

for melody in $melodies; do
    for i in $(seq 1 $range); do
        output_files="$output_files $basepath/$melody$i.txt"
    done
done

output_files="$output_files $basepath/baseline_melody_no_aug_1.txt"


echo "python scripts/output_comparison.py --truth temp/unaltered.txt --outputs $output_files --output-csv Logs/output_comparison.csv"

python scripts/output_comparison.py --truth temp/unaltered.txt --outputs $output_files --output-csv Logs/output_comparison.csv

rm -rf temp
```