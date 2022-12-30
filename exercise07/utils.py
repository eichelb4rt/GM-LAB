import numpy as np
import scipy.stats

def mean_confidence_interval(data:np.ndarray, confidence:float=0.95) -> tuple:
    '''
    Calculates the confidence interval for the mean of given data.
    Taken from this stackoverflow question:
        https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    
    @Params:
        data... array of sample points
        confidence... confidence of interval

    @Returns:
        confidence interval as a triple (mean, lower, upper)
    '''

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h