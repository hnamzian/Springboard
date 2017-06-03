import numpy as np
import scipy as scp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class hypothesisTest_oneSample(object):
    def __init__(self, data):
        self.sample_mean = np.mean(data)
        self.sample_std = np.std(data)
        self.sample_mean_std = np.std(data)/np.sqrt(len(data))
        self.sample_size = len(data)
        print 'sample size: %d' % self.sample_size
        print 'sample_mean: %f' % self.sample_mean
        print 'sample_std:  %f' % self.sample_std
        print 'SE(sample_mean std): %f' % self.sample_mean_std
    
    def CI_zTest(self, pcg):
        z = 0.5+(0.01*0.5*pcg)
        CI = (np.array([-1, 1]) * stats.norm.ppf(z) * self.sample_mean_std) + self.sample_mean
        ME = scp.stats.norm.ppf(z) * self.sample_mean_std
        return CI, ME
            
    def CI_tTest(self, pcg):
        t = 0.5+(0.01*0.5*pcg)
        df = self.sample_size-1
        CI = (np.array([-1 , 1]) * scp.stats.t.ppf(t, df) * self.sample_mean_std) + self.sample_mean
        ME = scp.stats.t.ppf(t, df) * self.sample_mean_std
        return CI, ME
    
    def pValue_zTest(self, mean_hypo):
        test_value = self.sample_mean
        z = (test_value - mean_hypo)/self.sample_mean_std
        pValue_oneTailed = 1-scp.stats.norm.cdf(np.abs(z))
        pValue_twoTailed = 2*(1-scp.stats.norm.cdf(np.abs(z)))
        return pValue_oneTailed, pValue_twoTailed
    
    def pValue_tTest(self, mean_hypo):
        test_value = self.sample_mean
        t = (test_value - mean_hypo)/self.sample_mean_std
        df = self.sample_size-1
        pValue_oneTailed = 1-scp.stats.t.cdf(np.abs(t), df)
        pValue_twoTailed = 2*(1-scp.stats.t.cdf(np.abs(t), df))
        return pValue_oneTailed, pValue_twoTailed
		


class DiffMeanhypothesisTest_twoSample(object):
    def __init__(self, data1, data2, legends):
        self.sample_mean_1 = np.mean(data1)
        self.sample_std_1 = np.std(data1)
        self.sample_mean_std_1 = np.std(data1)/np.sqrt(len(data1))
        self.sample_size_1 = len(data1)
        
        self.sample_mean_2 = np.mean(data2)
        self.sample_std_2 = np.std(data2)
        self.sample_mean_std_2 = np.std(data2)/np.sqrt(len(data2))
        self.sample_size_2 = len(data2)
        
        self.diff_sample_mean = (self.sample_mean_1 - self.sample_mean_2)
        self.diff_sample_mean_std = np.sqrt(self.sample_mean_std_1**2 + self.sample_mean_std_2**2)

        self.legends = legends
        
        print '%s sample size: %d'     % (self.legends[0], self.sample_size_1)
        print '%s sample_mean: %f'     % (self.legends[0], self.sample_mean_1)
        print '%s sample_std:  %f'     % (self.legends[0], self.sample_std_1)
        print '%s sample_mean std: %f' % (self.legends[0], self.sample_mean_std_1)
        
        print '%s sample size: %d'     % (self.legends[1], self.sample_size_2)
        print '%s sample_mean: %f'     % (self.legends[1], self.sample_mean_2)
        print '%s sample_std:  %f'     % (self.legends[1], self.sample_std_2)
        print '%s sample_mean std: %f' % (self.legends[1], self.sample_mean_std_2)
        
        print 'Difference Sample_mean: %f'     % self.diff_sample_mean
        print 'Difference Sample_mean Std: %f' % self.diff_sample_mean_std
    
    def CI_zTest(self, pcg):
        z = 0.5+(0.01*0.5*pcg)
        CI = (np.array([-1, 1]) * scp.stats.norm.ppf(z) * self.diff_sample_mean_std) + self.diff_sample_mean
        ME = scp.stats.norm.ppf(z) * self.diff_sample_mean_std
        return CI, ME
        
    def CI_tTest(self, pcg):
        t = 0.5+(0.01*0.5*pcg)
        # df is calculated based upon std_ratio (pooled or unpooled)
        # link: https://onlinecourses.science.psu.edu/stat200/node/60
        std_ratio = self.sample_mean_std_1/self.sample_mean_std_2
        if (std_ratio<2 and std_ratio>0.5):
            df = self.sample_size_1 + self.sample_size_2-2
        else:
            df = (self.diff_sample_mean_std**4)/ \
            (((self.sample_mean_std_1**4)/self.sample_size_1) +  \
             ((self.sample_mean_std_2**4)/self.sample_size_2))
        CI = (np.array([-1 , 1]) * scp.stats.t.ppf(t, df) * self.diff_sample_mean_std) + self.diff_sample_mean
        ME = scp.stats.t.ppf(t, df) * self.diff_sample_mean_std
        return CI, ME
    
    def pValue_zTest(self, diff_mean_hypo):
        test_diff_value = self.sample_mean_1 - self.sample_mean_2
        z = (test_diff_value - diff_mean_hypo)/self.diff_sample_mean_std
        pValue_oneTailed = 1-scp.stats.norm.cdf(np.abs(z))
        pValue_twoTailed = 2*(1-scp.stats.norm.cdf(np.abs(z)))
        return pValue_oneTailed, pValue_twoTailed
    
    def pValue_tTest(self, diff_mean_hypo):
        test_diff_value = self.sample_mean_1 - self.sample_mean_2
        t = (test_diff_value - diff_mean_hypo)/self.diff_sample_mean_std
        std_ratio = self.sample_mean_std_1/self.sample_mean_std_2
        if (std_ratio<2 and std_ratio>0.5):
            df = self.sample_size_1 + self.sample_size_2-2
        else:
            df = (self.diff_sample_mean_std**4)/ \
            (((self.sample_mean_std_1**4)/self.sample_size_1) +  \
             ((self.sample_mean_std_2**4)/self.sample_size_2))
        pValue_oneTailed = 1-scp.stats.t.cdf(np.abs(t), df)
        pValue_twoTailed = 2*(1-scp.stats.t.cdf(np.abs(t), df))
        return pValue_oneTailed, pValue_twoTailed
		
		