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
def expectation_maximization(df):
    data = np.array(df['x'])
    
    # Initialize the parameters
    mu1 = np.mean(data)
    mu2 = np.mean(data) * 2
    lambda1 = 1
    lambda2 = 1
    n = len(data)
    
    # Implement the EM algorithm
    for i in range(100):
        # Expectation step
        p1 = stats.poisson.pmf(data, mu1)
        p2 = stats.poisson.pmf(data, mu2)
        w1 = lambda1 * p1 / (lambda1 * p1 + lambda2 * p2)
        w2 = lambda2 * p2 / (lambda1 * p1 + lambda2 * p2)
        
        # Maximization step
        lambda1 = np.sum(w1) / n
        lambda2 = np.sum(w2) / n
        mu1 = np.sum(w1 * data) / (lambda1 * n)
        mu2 = np.sum(w2 * data) / (lambda2 * n)
    
    print("lambda1 = ", lambda1)
    print("lambda2 = ", lambda2)

df = pd.read_csv('kmer_depth.csv')

# Plot a histogram
hist = ggplot(df, aes(x='x')) + geom_histogram(binwidth=1, fill='#0072B2', color='black') + labs(x='k-mer Depth', y='Count')
print(hist)

# Plot a density plot
density = ggplot(df, aes(x='x')) + geom_density(fill='#0072B2', color='black') + labs(x='k-mer Depth', y='Density')
print(density)

# running your EM implementation
expectation_maximization(df)