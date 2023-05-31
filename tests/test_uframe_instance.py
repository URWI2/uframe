import unittest
import numpy as np 
import scipy
from uframe_instance import uframe_instance

class Testuframe_insatnce(unittest.TestCase):
    def test_scipy_kde(self):
        m1 = np.random.normal(size=200)
        m2 = np.random.normal(scale=0.5, size=200)
        
        kernel = scipy.stats.gaussian_kde(np.vstack([m1,m2]))      
        
        
        instance =  uframe_instance(kernel,None, [0,1,[]])


if __name__ == '__main__':
    unittest.main()


