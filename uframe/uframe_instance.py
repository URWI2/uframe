"""
uframe_instance class used in uframe to depict a single uncertain data instance.

"""
import scipy 
import numpy as np
import sklearn
from sklearn.neighbors import KernelDensity
import warnings


class uframe_instance(): 
    """
   A class used to represent an singel uncertain data instance.

   ...

   Attributes
   ----------
   continuous : class
       A class describing the underlying uncertainty.
   certain_data : np.array 
       Numpy array of certain values
   indices : [list,list]
       associations of indices to continuous and certain

   Methods
   -------
   sample(n = 1, seed = None)
       Samples n samples from the uncertain instance
       
   """
   
    def __init__(self,uncertain_obj:scipy.stats.gaussian_kde, certain_data:np.array, indices:list[list]): 
        """
        Parameters
        ----------
        uncertain_obj : scipy.stats._kde.gaussian_kde | scipy.stats
            Scipy Kernel Density Estimation or other, that describes the ucnertain Variables of the instance.
        certain_data 1D np.array
            Certain data instances.
        indices : tupel
            tupel of indices which indicates the order in which samples and modal values should be returned.
        """
        if uncertain_obj is not None:
            assert  type(uncertain_obj) in [scipy.stats._kde.gaussian_kde, sklearn.neighbors._kde.KernelDensity]
        if certain_data is not None:
            assert type(certain_data) in [np.ndarray,np.array]
            assert len(certain_data) == len(indices[1])
        assert self.__check_indices(uncertain_obj, certain_data, indices)
        
        
        
        if type(uncertain_obj) == scipy.stats._kde.gaussian_kde:
            self.__init_scipy_kde(uncertain_obj)
        if type(uncertain_obj) == sklearn.neighbors._kde.KernelDensity: 
            self.__init_sklearn_kde(uncertain_obj)
            
            
            
        self.certain_data = certain_data 
        self.indices = indices
        self.n_vars = self.__get_len(indices)
        
    def sample(self,n: int = 1, seed: int= None): 
        
        if type(self.continuous) == scipy.stats._kde.gaussian_kde:
            return self.__align(self.__sample_scipy_kde(n))
        
        if type(self.continuous) == sklearn.neighbors._kde.KernelDensity:
            return self.__align(self.__sample_sklearn_kde(n, seed = seed))
        
        
    def modal(self): 
        
        if type(self.continuous) == scipy.stats._kde.gaussian_kde:
            return self.__align(self.__modal_scipy_kde())
        
        if type(self.continuous) == sklearn.neighbors._kde.KernelDensity:
            return self.__align(self.__modal_sklearn_kde())
        
        
        
        
    def __str__(self):
        print("Data Instance")
        
    def __repr__(self):
        print("Data Instance")
    
    def __len__(self): 
        return self.n_vars
        
    def __modal_scipy_kde(self): 
        opt = scipy.optimize.basinhopping(lambda x: -self.continuous.pdf(x),np.zeros(self.n_vars) )
        return opt.x
        
    def __modal_sklearn_kde(self): 
        opt = scipy.optimize.basinhopping(lambda x: -self.continuous.score_samples(x.reshape(1,-1)), np.zeros(len(self.indices[0])) )
        return opt.x
        
        
    def __init_scipy_kde(self, kernel):
        self.continuous = kernel
    
    def __init_sklearn_kde(self, kernel): 
        self.continuous = kernel
        
        if self.continuous.get_params()["kernel"] not in ["gaussian", "tophat"]: 
            warnings.warn("The provided KDE does not has an gaussian or tophat kernel, this might result in Errors")
    
    
    def __sample_scipy_kde(self, n): 
        return self.continuous.resample(n).transpose()
    
    def __sample_sklearn_kde(self, n, seed = None) :
        return self.continuous.sample(n_samples = n, random_state = seed)
    
    
    def __get_len(self, indices): 
        if len(indices[0])==0: 
            return max(self.indices[1])+1
                       
        if len(indices[1])==0: 
            return max(self.indices[0])+1
        
        return max(max(self.indices[0]),max(self.indices[0])) +1
    
    def __check_indices(self, uncertain_obj, certain_data, indices):
        assert type(indices) == list
        assert len(indices) == 2
        
        if len(indices[1]) == 0:
            assert certain_data == None
        
        if certain_data == None: 
            assert len(indices[1]) == 0 
        
        if len(indices[0]) == 0 and uncertain_obj == None:
            return True
        
        if type(uncertain_obj) == scipy.stats._kde.gaussian_kde:
            return len(indices[0]) == uncertain_obj.dataset.shape[0]
        
        if type (uncertain_obj) == sklearn.neighbors._kde.KernelDensity:
            return len(indices[0]) == uncertain_obj.n_features_in_
        
    
    def __align(self,uncertain_values): 
        
        ret = np.zeros(self.n_vars)
        ret[self.indices[0]] = np.array(uncertain_values)
        ret[self.indices[1]] = self.certain_data
        return ret
    
