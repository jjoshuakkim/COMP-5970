import numpy as np
import pandas as pd
import scipy.stats as stats
from plotnine import *
'''
Homework 3 problem 9 -- expectation maximization
implement expectation/maximization of 2 poisson distributions.
Output the lambda values for each distribution
feel free to make helper functions
'''
df = pd.read_csv('kmer_depth.csv')

def expectation(data, m1, m2, lambda1, lambda2):
    p1 = stats.poisson.pmf(data, m1)       # estimating the probability distribution
    p2 = stats.poisson.pmf(data, m2)
    w1 = lambda1 * p1 / (lambda1 * p1 + lambda2 * p2)       # Uses the prob estimates to calculate the weight for each dist using Bayes
    w2 = lambda2 * p2 / (lambda1 * p1 + lambda2 * p2)
    return w1, w2

def maximization(data, w1, w2):
    n = len(data)
    lambda1 = np.sum(w1) / n                # Paramters on the poisson is updated using the weights
    lambda2 = np.sum(w2) / n
    mu1 = np.sum(w1 * data) / (lambda1 * n)
    mu2 = np.sum(w2 * data) / (lambda2 * n)
    return mu1, mu2, lambda1, lambda2

def expectation_maximization(df):
    data = np.array(df['x'])
    
    # Initialize the parameters
    mean1 = np.mean(data)             # Means of the distribution to model data
    mean2 = np.mean(data) * 2 
    lambda1 = 1
    lambda2 = 1
    
    for i in range(100):
        w1, w2 = expectation(data, mean1, mean2, lambda1, lambda2)
        mean1, mean2, lambda1, lambda2 = maximization(data, w1, w2)
    
    print("lambda1 = ", mean1)
    print("lambda2 = ", mean2)

def graph(list):
    hist = ggplot(list, aes(x='x')) + geom_histogram(binwidth=1, fill='#6e0d27', color='black') + labs(x='k-mer Depth', y='Count')
    density = ggplot(list, aes(x='x')) + geom_density(fill='#6e0d27', color='black') + labs(x='k-mer Depth', y='Density')
    print(hist) # Histogram
    print(density) # Density

# running your EM implementation
expectation_maximization(df)
graph(df)
