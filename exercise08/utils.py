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

# In case you want to build everything from scratch

class MT_19937():
    def __init__(self, seed:int):
        # general params
        self.w = 32 # word length -> range of resulting numbers
        self.n = 624 # number of recurrence

        # initialization
        self.seed = seed
        self.f = 1812433253

        # twist
        self.m = 397 # offset used in recurrence
        self.r = 1 # seperation point of bit sequence
        self.a = 0x9908b0df # used for twist if last bit = 1
        
        # tempering
        self.u = 11
        self.s = 7
        self.b = 0x9D2C5680
        self.t = 15
        self.c = 0xEFC60000
        self.l = 18
        
        # states
        self.index = 0
        self.state = None
        self.init_state()
        
    def sample(self) -> float:
        '''
        Samples a float in [0, 1].
        '''
        if self.index == 0:
            self.twist()

        number = self.tempering(self.state[self.index])
        number = number/(2**self.w - 1)
        
        self.index = (self.index + 2)%self.n
        return number
                
    def to_intw(self, number:int) -> int:
        return int((2**self.w-1) & number)

    def combine(self, number1:int, number2:int) -> int:
        mask2 = 2**(self.w-self.r)-1 # ...00001111...
        mask1 = (2**(self.w)-1) - mask2 # ...11110000...
        return self.to_intw((number1&mask1) + (number2&mask2))
        
    def init_state(self):
        self.state = [0]*(self.n)
        self.state[0] = self.seed
        for i in range(1, self.n):
            self.state[i] = self.to_intw(self.f*(self.state[i-1]^(self.state[i-1]>>(self.w-2))) + i)
        
    def twist(self):
        for i in range(self.n):
            number1 = self.state[i]
            number2 = self.state[(i+1) % self.n]
            number3 = self.state[(i+self.m) % self.n]
            
            comb_number = self.combine(number1, number2)
            comb_shift = comb_number >> 1
            if comb_number%2 == 1:
                comb_shift = comb_shift^self.a
            new_number = comb_shift^number3
            self.state[i] = new_number
            
    def tempering(self, number:int) -> int:
        number = number^(number >> self.u)
        number = number^((number << self.s) & self.b)
        number = number^((number << self.t) & self.c)
        number = number^(number >> self.l)
        return self.to_intw(number)