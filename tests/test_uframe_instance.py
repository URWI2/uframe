import unittest
import numpy as np 
import scipy
import sklearn
from uframe_instance import uframe_instance

class Testuframe_insatnce(unittest.TestCase):
    def test_scipy_kde(self):
        m1 = np.random.normal(size=200)
        m2 = np.random.normal(scale=0.5, size=200)
        
        kernel = scipy.stats.gaussian_kde(np.vstack([m1,m2]))      

        instance_0 =  uframe_instance(kernel,np.array([1,1]), [[0,3],[1,2]])
        instance_1 =  uframe_instance(kernel,None, [[0,1],[]])
        instance_2 =  uframe_instance(None,np.array([1,1]), [[],[0,1]])
        
        for instance in [instance_0,instance_1,instance_2]:
            self.instance_check(instance)
            
    def test_sklearn_kde(self): 
        m1 = np.random.normal(size=200)
        m2 = np.random.normal(scale=0.5, size=200)
        
        kernel = sklearn.neighbors.KernelDensity().fit(np.vstack([m1,m2]).transpose())      

        instance_0 =  uframe_instance(kernel,np.array([1,1]), [[0,3],[1,2]])
        instance_1 =  uframe_instance(kernel,None, [[0,1],[]])
        instance_2 =  uframe_instance(None,np.array([1,1]), [[],[0,1]])
        
        for instance in [instance_0,instance_1,instance_2]:
            self.instance_check(instance)
        
    def test_scipy_rv_contuinuous(self): 
        

        for rv in [scipy.stats.gamma(2), scipy.stats.norm(), scipy.stats.alpha(2),
                   scipy.stats.anglit(), scipy.stats.arcsine(),scipy.stats.argus(1),
                   scipy.stats.cauchy(), scipy.stats.chi(1), scipy.stats.expon(),
                   scipy.stats.exponnorm(1),scipy.stats.uniform(1), scipy.stats.logistic(),
                   scipy.stats.wald(1),scipy.stats.lognorm(0.2), scipy.stats.loguniform(0.02, 1.25),
                   scipy.stats.laplace()]:
            self.check_single_rv(rv)
        
        
    def check_single_rv(self, rv): 
        
        instance_0 =  uframe_instance(rv,np.array([1,1]), [[1],[0,2]])
        instance_1 =  uframe_instance(rv,None, [[0],[]])
        instance_2 =  uframe_instance(None,np.array([1,1]), [[],[0,1]])

        for instance in [instance_0,instance_1,instance_2]:
            self.instance_check(instance)
    
    def test_scipy_rv_multivariate(self): 
        
        
        for rv in [scipy.stats.multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])]:
            self.check_multivariate_rv(rv)
            
            
            
    def check_multivariate_rv(self, rv):
        
        instance_0 =  uframe_instance(rv,np.array([1,1]), [[1,3],[0,2]])
        instance_1 =  uframe_instance(rv,None, [[0,1],[]])
        instance_2 =  uframe_instance(None,np.array([1,1]), [[],[0,1]])

        for instance in [instance_0,instance_1,instance_2]:
            self.instance_check(instance)
    
    

    def test_dist_list(self): 
        rv = [scipy.stats.norm(10), scipy.stats.norm(2)]       
        
        instance_0 =  uframe_instance(rv,np.array([1,1]), [[0,3],[1,2]])
        instance_1 =  uframe_instance(rv,None, [[0,1],[]])
        instance_2 =  uframe_instance(None,np.array([1,1]), [[],[0,1]])

        instance = instance_0        
        for instance in [instance_0,instance_1,instance_2]:
            self.instance_check(instance)
          

    def instance_check(self,instance): 
        instance.mode()
        instance.sample(1)
        instance.sample(10)

    
        

if __name__ == '__main__':
    unittest.main()


