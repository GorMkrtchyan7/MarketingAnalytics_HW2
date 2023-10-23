#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from logs import *
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        self.p = p
        self.num_actions = len(p)
        self.q_true = np.array(p)
        self.q_estimates = np.zeros(self.num_actions)
        self.action_counts = np.zeros(self.num_actions)
        self.rewards = []
        self.cumulative_rewards = []

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        action = self.select_action()
        reward = self.q_true[action] + np.random.normal(0, 1)
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]
        self.rewards.append(reward)
        self.cumulative_rewards.append(sum(self.rewards))


    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self, num_trials):
        for _ in range(num_trials):
            self.pull()

    @abstractmethod
    def report(self, algorithm_name):
        # Create a Pandas DataFrame for easier data analysis and export to CSV
        data = {'Action': list(range(self.num_actions)),
                'Counts': self.action_counts.tolist(),
                'Estimated Q-values': self.q_estimates.tolist()}
        df = pd.DataFrame(data)
        df.to_csv(f'{algorithm_name}_results.csv', index=False)

        average_reward = np.mean(self.rewards)
        cumulative_reward = sum(self.rewards)
        cumulative_regret = max(self.q_true) * len(self.rewards) - cumulative_reward

        logger.info(f'{algorithm_name} - Average Reward: {average_reward:.2f}')
        logger.info(f'{algorithm_name} - Cumulative Reward: {cumulative_reward:.2f}')
        logger.info(f'{algorithm_name} - Cumulative Regret: {cumulative_regret:.2f}')

#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, p, epsilon):
        super().__init__(p)
        self.epsilon = epsilon

    def select_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_estimates)

    def pull(self):
        action = self.select_action()
        reward = self.q_true[action] + np.random.normal(0, 1)
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]
        self.rewards.append(reward)
        self.cumulative_rewards.append(sum(self.rewards))

    # Implement the missing abstract methods
    def __repr__(self):
        return f'EpsilonGreedy Bandit with epsilon={self.epsilon}'

    def experiment(self, num_trials):
        for _ in range(num_trials):
            self.pull()

    def update(self):
        pass

    def report(self):
        pass

    
class ThompsonSampling(Bandit):
    def __init__(self, p, precision=1.0):
        super().__init__(p)
        self.precision = precision

    def select_action(self):
        theta_samples = np.random.normal(self.q_estimates, scale=1.0 / (self.precision * self.action_counts))
        return np.argmax(theta_samples)

    def pull(self):
        action = self.select_action()
        reward = self.q_true[action] + np.random.normal(0, 1)
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]
        self.rewards.append(reward)
        self.cumulative_rewards.append(sum(self.rewards))

    # Implement the missing abstract methods
    def __repr__(self):
        return f'Thompson Sampling Bandit with precision={self.precision}'

    def experiment(self, num_trials):
        for _ in range(num_trials):
            self.pull()

    def update(self):
        pass

    def report(self):
        pass

    


class Visualization():

    def plot1(self):
        plt.plot(self.cumulative_rewards)
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Reward')
        plt.title(f'{self.__class__.__name__} - Learning Process')
        plt.show()


    def plot2(self, epsilon_greedy_bandit, thompson_sampling_bandit, num_trials):
    # Calculate cumulative regrets for both bandits
        cumulative_regrets_eg = [max(epsilon_greedy_bandit.q_true) * i - sum(epsilon_greedy_bandit.rewards[:i]) for i in range(1, num_trials + 1)]
        cumulative_regrets_ts = [max(thompson_sampling_bandit.q_true) * i - sum(thompson_sampling_bandit.rewards[:i]) for i in range(1, num_trials + 1)]

        # Create a subplot with two bar charts
        fig, axs = plt.subplots(2, 1, sharex=True)

        # Plot cumulative rewards
        x_values = range(1, num_trials + 1)  # Ensure x_values match the number of trials
        axs[0].bar(x_values, epsilon_greedy_bandit.cumulative_rewards, label='Epsilon Greedy', alpha=0.7)
        axs[0].bar(x_values, thompson_sampling_bandit.cumulative_rewards, label='Thompson Sampling', alpha=0.7)
        axs[0].set_ylabel('Cumulative Reward')
        axs[0].legend(loc='upper left')

        # Plot cumulative regrets
        x_values = range(1, num_trials)  # Ensure x_values match the number of trials
        axs[1].bar(x_values, cumulative_regrets_eg, label='Epsilon Greedy', alpha=0.7)
        axs[1].bar(x_values, cumulative_regrets_ts, label='Thompson Sampling', alpha=0.7)
        axs[1].set_xlabel('Trials')
        axs[1].set_ylabel('Cumulative Regret')
        axs[1].legend(loc='upper left')

        plt.show()



def comparison():
    # think of a way to compare the performances of the two algorithms VISUALLY and 
    pass

if __name__ == '__main':
    Bandit_Rewards = [1, 2, 3, 4]
    NumberOfTrials = 20000

    # Create bandits for each algorithm
    epsilon_greedy_bandit = EpsilonGreedy(Bandit_Rewards, epsilon=0.1)
    thompson_sampling_bandit = ThompsonSampling(Bandit_Rewards, precision=0.1)

    epsilon_greedy_bandit.plot2(epsilon_greedy_bandit, thompson_sampling_bandit)

    # Run experiments
    epsilon_greedy_bandit.experiment(NumberOfTrials)
    thompson_sampling_bandit.experiment(NumberOfTrials)

    # Report results and visualize learning process
    epsilon_greedy_bandit.report('EpsilonGreedy')
    thompson_sampling_bandit.report('ThompsonSampling')
    epsilon_greedy_bandit.plot1()
    thompson_sampling_bandit.plot1()
    num_trials = NumberOfTrials  
    epsilon_greedy_bandit.plot2(epsilon_greedy_bandit, thompson_sampling_bandit, num_trials)

    # Store the rewards in a CSV file
    rewards_data = {
        'Bandit': ['EpsilonGreedy'] * NumberOfTrials + ['ThompsonSampling'] * NumberOfTrials,
        'Reward': epsilon_greedy_bandit.rewards + thompson_sampling_bandit.rewards,
        'Algorithm': ['EpsilonGreedy'] * NumberOfTrials + ['ThompsonSampling'] * NumberOfTrials
    }
    df_rewards = pd.DataFrame(rewards_data)
    df_rewards.to_csv('rewards.csv', index=False)

    # Print cumulative reward and regret
    cumulative_reward = sum(epsilon_greedy_bandit.rewards) + sum(thompson_sampling_bandit.rewards)
    cumulative_regret = max(Bandit_Rewards) * (2 * NumberOfTrials) - cumulative_reward
    logger.info(f'Total Cumulative Reward: {cumulative_reward:.2f}')
    logger.info(f'Total Cumulative Regret: {cumulative_regret:.2f}')
    

