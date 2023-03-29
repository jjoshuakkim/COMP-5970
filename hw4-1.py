import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from scipy.special import logsumexp
from plotnine import *
import plotnine as pn
import matplotlib.pyplot as plt

'''
Homework 4 problem 1 -- Plot data (please save to file, dont just print it)
plot the timeseries data for simulated nanopore
'''
plt.rcParams["figure.figsize"] = [15, 10]

def plot_timeseries_data(data):
    data.plot.scatter(x='time',y='level')
    plt.savefig('nanospore_plot.png')
'''
Homework 4 problem 2
What is the approximate duration of each "event" in this data given this plot?
'''
approx_duration = 50

'''
Homework 4 problem 3 -- HMM maximum likelihood state sequence with 4 states
state 1 - T corresponds to a normal distribution with mean 100 and sd 15
state 2 - A corresponds to a normal dist with mean 150 and sd 25
state 3 - G correcponds to a normal dist with mean 300 and sd 50
state 4 - C corresponds to a normal dist with mean 350 and sd 25
transitions between states are 1/50 and transitions to same state is 49/50
'''
def HMM_MLE(df):
    states = ['T', 'A', 'G', 'C']           # Initialize states and transition probabilities
    num_states = len(states)

    start_prob = np.log(np.ones(num_states) / num_states)       # Uniform prior probabilities for each state and between states
    trans_prob = np.log(np.full((num_states, num_states), 1/50))
    np.fill_diagonal(trans_prob, np.log(49/50))         # Higher prob of staying in same state
    mean = np.array([100, 150, 300, 350])
    sd = np.array([15, 25, 50, 25])

    num_obs = df.shape[0]           
    log_alpha = np.zeros((num_obs, num_states))         # Forward Backward implementation
    log_beta = np.zeros((num_obs, num_states))
    log_alpha[0] = start_prob + norm.logpdf(df['level'][0], loc=mean, scale=sd)
    for t in range(1, num_obs):
        log_alpha[t] = norm.logpdf(df['level'][t], loc=mean, scale=sd) + \
                       np.max(log_alpha[t-1] + trans_prob.T, axis=1)

    log_beta[-1] = np.zeros(num_states)
    for t in range(num_obs-2, -1, -1):
        log_beta[t] = np.max(log_beta[t+1] + norm.logpdf(df['level'][t+1], loc=mean, scale=sd) + trans_prob, axis=1)

    log_gamma = log_alpha + log_beta - np.max(log_alpha[-1])            # This computes the log prob of being in each state at each time point
    inferred_states = []                    # Determines most likely state sequence given the computed log prob
    for t in range(num_obs):
        state_index = np.argmax(log_gamma[t])
        state_name = states[state_index]
        inferred_states.append(state_name)  

    return np.array(inferred_states)

'''
Homework 4 problem 4
plot output of problem 3. Here, please make 1 plot with 4 plots overlayed with different colors.
'''
def plot_MLE(state_sequence):
    states = ['T', 'A', 'G', 'C']
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728']
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, state in enumerate(states):
        state_mask = (state_sequence == state)
        ax.scatter(df['time'][state_mask], df['level'][state_mask], s=5, color=colors[i], label=state)
    ax.set_xlabel('Time')
    ax.set_ylabel('Level')
    ax.set_title('Maximum Likelihood Estimation')
    ax.legend()
    plt.savefig('plot_mle.png', dpi=300)

'''
Homework 4 problem 5
Give the most likely sequence this data corresponds to given the likely 
event length you found from plotting the data
print this sequence of A/C/G/Ts
'''
def MLE_seq(df, event_length):
    num_events = df.shape[0] // event_length
    observations = df['level'].values
    events = np.array_split(observations, num_events)
    sequence = ''
    for event in events:
        event_df = pd.DataFrame({'time': df['time'][:len(event)], 'level': event})
        event_seq = HMM_MLE(event_df)
        sequence += ''.join(event_seq)
    print(sequence)

'''
Homework 4 problem 6
Forward/backward algorithm giving posterior probabilities for each time point for each level
'''
def HMM_posterior(df):
    states = ['T', 'A', 'G', 'C']
    num_states = len(states)
    start_prob = np.full(num_states, 1/num_states)          # Initializes prior and trans probs between states
    trans_prob = np.full((num_states, num_states), 1/num_states)
    np.fill_diagonal(trans_prob, (num_states-1)/num_states)
    mean = np.array([100, 150, 300, 350])
    sd = np.array([15, 25, 50, 25])

    observations = df['level'].values
    log_likelihoods = norm.logpdf(observations[:, np.newaxis], loc=mean, scale=sd)      # Calc likelihoods in logspace of each state's obs dist
    log_forward = np.zeros((len(observations), num_states))
    log_backward = np.zeros((len(observations), num_states))
    log_forward[0] = np.log(start_prob) + log_likelihoods[0]

    for t in range(1, len(observations)):       # Calc forward prob
        log_forward[t] = np.logaddexp.reduce(log_forward[t-1, :] + np.log(trans_prob), axis=1) + log_likelihoods[t]

    log_backward[-1] = 0
    for t in range(len(observations)-2, -1, -1):        # Calc backward prob
        log_backward[t] = np.logaddexp.reduce(log_backward[t+1, :] + log_likelihoods[t+1] + np.log(trans_prob), axis=1)

    log_posteriors = log_forward + log_backward
    log_evidence = np.logaddexp.reduce(log_posteriors[0])
    log_posteriors -= log_evidence
    posteriors = np.exp(log_posteriors)

    return posteriors


'''
Homework 4 problem 7
plot output of problem 5, this time, plot with 4 facets using facet_wrap
'''
def plot_posterior(posteriors):
    states = ['T', 'A', 'G', 'C']
    time = range(len(posteriors))
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    axs = axs.flatten()

    for i, state in enumerate(states):
        axs[i].plot(time, posteriors[:, i], color='C{}'.format(i), label=state)
        axs[i].set_title(state)
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Probability')
        axs[i].legend()

    plt.savefig('plot_posteriors.png')


df = pd.read_csv("nanopore.csv")
plot_timeseries_data(df)
state_sequence = HMM_MLE(df)
plot_MLE(state_sequence)
MLE_seq(df, approx_duration)
posteriors = HMM_posterior(df)
plot_posterior(posteriors)