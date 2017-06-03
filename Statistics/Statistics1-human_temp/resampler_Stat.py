import numpy as np
import scipy as scp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

#This is a modified version of Allen Downey's code
class Resampler_new(object):
    #Represents a framework for computing sampling distributions."""
    
    def __init__(self, samples, xlim=None, legend='', title=''):
        #Stores the actual sample."""
        self.samples = samples
        self.n = [len(sample) for sample in samples]
        self.num_of_samples = len(samples)
        self.xlim = xlim
        self.legend = legend
        self.title = title
        
    def resample(self):
        #Generates a new sample by choosing from the original
        #sample with replacement.
        
        new_samples = [np.random.choice(sample, len(sample), replace=True) \
                       for sample in self.samples]
        return new_samples
    
    def sample_stat(self, samples):
        #Computes a sample statistic using the original sample or a
        #simulated sample.
        
        return [sample.mean() for sample in samples]
    
    def compute_sample_statistics(self, iters=1000):
        #Simulates many experiments and collects the resulting sample
        #statistics.
        
        stats = [self.sample_stat(self.resample()) for i in range(iters)]
        return np.array(stats)
    
    def summarize_sampling_distribution(self, sample_stats, pcg):
        return sample_stats.mean(axis=0), \
               sample_stats.std(axis=0), \
               np.percentile(sample_stats, [100-pcg, pcg], axis=0)
    
    def plot_sample_stats(self, pcg):
        #Runs simulated experiments and summarizes the results.
        
        sample_stats = self.compute_sample_statistics()
        [sample_stats_mean, sample_stats_std, sample_stats_CI] = self.summarize_sampling_distribution(sample_stats, pcg)
        for i in range(self.num_of_samples):
            print '%s sample_stats mean: %f'    % (self.legend[i], sample_stats_mean[i])
            print '%s sample_stats std:  %f'    % (self.legend[i], sample_stats_std[i])
            print '%s %d%% sample_stats mean: ' % (self.legend[i], pcg) + str(sample_stats_CI[i])
            sns.distplot(sample_stats[:,i], norm_hist=False)
        plt.xlabel('sample statistic')
        plt.ylabel('Probability')
        plt.title(self.title)
        if self.xlim is not None: plt.xlim(self.xlim)
        plt.legend(self.legend)
		
		
#This piece of code is taken from Allen Downey's
class Resampler(object):
    """Represents a framework for computing sampling distributions."""
    
    def __init__(self, sample, xlim=None, legend='', title=''):
        #Stores the actual sample."""
        self.sample = sample
        self.n = len(sample)
        self.xlim = xlim
        self.legend = legend
        self.title = title
        
    def resample(self):
        #Generates a new sample by choosing from the original
        #sample with replacement.
        
        new_sample = np.random.choice(self.sample, self.n, replace=True)
        return new_sample
    
    def sample_stat(self, sample):
        #Computes a sample statistic using the original sample or a
        #simulated sample.
        
        return sample.mean()
    
    def compute_sample_statistics(self, iters=1000):
        #Simulates many experiments and collects the resulting sample
        #statistics.
        
        stats = [self.sample_stat(self.resample()) for i in range(iters)]
        return np.array(stats)
    
    def summarize_sampling_distribution(self, sample_stats, pcg):
        return sample_stats.mean(), sample_stats.std(), np.percentile(sample_stats, [100-pcg, pcg])
    
    def plot_sample_stats(self, pcg):
        #Runs simulated experiments and summarizes the results.
        
        sample_stats = self.compute_sample_statistics()
        [sample_stats_mean, sample_stats_std, sample_stats_CI] = self.summarize_sampling_distribution(sample_stats, pcg)
        print '%s sample_stats mean: %f'    % (self.legend, sample_stats_mean)
        print '%s sample_stats std:  %f'    % (self.legend, sample_stats_std)
        print '%s %d%% sample_stats mean: ' % (self.legend, pcg) + str(sample_stats_CI)
        sns.distplot(sample_stats, norm_hist=False)
        plt.xlabel('sample statistic')
        plt.ylabel('Probability')
        plt.title(self.title)
        if self.xlim is not None: plt.xlim(self.xlim)
        plt.legend([self.legend])