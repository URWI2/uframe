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
           associations of indices to continuous and certain data
    
       Methods
       -------
       sample(n = 1, seed = None)
           Samples n samples from the uncertain instance
           
       mode()
           Uses an optimizer to finde the mode value of this instance
           
    """
   
    def __init__(self,uncertain_obj:scipy.stats.gaussian_kde, certain_data:np.array, indices:list[list]): 
        """
        Parameters
        ----------
        uncertain_obj : scipy.stats._kde.gaussian_kde | scipy.stats
            Scipy Kernel Density Estimation or other, that describes the ucnertain Variables of the instance.
        certain_data 1D np.array
            Certain data instances.
        indices : list
            list of indices which indicates the order in which samples and mode values should be returned.
        """
        
        self.certain_data = certain_data 
        self.indices = indices
        self.n_vars = self.__get_len(indices)
        
        if uncertain_obj is not None:
            assert (type(uncertain_obj) in [scipy.stats._kde.gaussian_kde, sklearn.neighbors._kde.KernelDensity,scipy.stats.multivariate_normal] or
                    issubclass(type(uncertain_obj), scipy.stats.rv_continuous) or 
                    issubclass(type(uncertain_obj), scipy.stats._multivariate.multi_rv_generic) or
                    issubclass(type(uncertain_obj), scipy.stats._multivariate.multi_rv_frozen) or 
                    issubclass(type(uncertain_obj), scipy.stats._distn_infrastructure.rv_continuous_frozen))
                    
        if certain_data is not None:
            assert type(certain_data) in [np.ndarray,np.array]
            assert len(certain_data) == len(indices[1])
        
        assert self.__check_indices(uncertain_obj, certain_data, indices)
        
        
        
        if type(uncertain_obj) == scipy.stats._kde.gaussian_kde:
            self.__init_scipy_kde(uncertain_obj)
        if type(uncertain_obj) == sklearn.neighbors._kde.KernelDensity: 
            self.__init_sklearn_kde(uncertain_obj)
        if (issubclass(type(uncertain_obj), scipy.stats.rv_continuous) or 
            issubclass(type(uncertain_obj), scipy.stats._distn_infrastructure.rv_continuous_frozen)or 
            issubclass(type(uncertain_obj), scipy.stats._multivariate.multi_rv_generic) or
            issubclass(type(uncertain_obj), scipy.stats._multivariate.multi_rv_frozen) ):
            self.__init_scipy_rv_c(uncertain_obj)
        
    
        
    def ev(self): 
        return("pending")
        
    def __str__(self):
        return("Data Instance")
        
    def __repr__(self):
        return("Data Instance")
    
    def __len__(self): 
        return self.n_vars
       
    #Initialization functions
    def __init_scipy_kde(self, kernel):
        self.continuous = kernel
        self.sample = self.__sample_scipy_kde
        self.mode = self.__mode_scipy_kde
        
    def __init_sklearn_kde(self, kernel): 
        self.continuous = kernel
        self.sample = self.__sample_sklearn_kde    
        self.mode = self.__mode_sklearn_kde
        
        if self.continuous.get_params()["kernel"] not in ["gaussian", "tophat"]: 
            warnings.warn("The provided KDE does not has an gaussian or tophat kernel, this might result in Errors")
            delattr(self, "sample")
    
    def __init_scipy_rv_c(self, rv): 
        self.continuous = rv 
        self.sample = self.__sample_scipy_rv_c
        self.mode = self.__mode_scipy_rv_c
    
    
    #mode functions
    def mode(self): 
        if not hasattr(self, "continuous"):
            return self.certain_data
    
    def __mode_scipy_kde(self): 
        opt = scipy.optimize.basinhopping(lambda x: -self.continuous.pdf(x),np.zeros(len(self.indices[0])) )
        return opt.x.reshape(1,-1)
        
    def __mode_sklearn_kde(self): 
        opt = scipy.optimize.basinhopping(lambda x: -self.continuous.score_samples(x.reshape(1,-1)), np.zeros(len(self.indices[0])) )
        return opt.x.reshape(1,-1)
    
    def __mode_scipy_rv_c(self): 
        if (issubclass(type(self.continuous), scipy.stats._multivariate.multi_rv_generic) or
            issubclass(type(self.continuous), scipy.stats._multivariate.multi_rv_frozen) ):
        
            opt = scipy.optimize.basinhopping(lambda x: -self.continuous.pdf(x), self.continuous.mean)
            return opt.x.reshape(1,-1)
            
        opt = scipy.optimize.basinhopping(lambda x: -self.continuous.pdf(x), self.continuous.mean())
        return opt.x.reshape(1,-1)
              
        
    
    #sampling functions
    def sample(self,n: int = 1, seed: int= None): 
        if not hasattr(self, "continuous"):
            return self.certain_data

    def __sample_scipy_kde(self, n): 
        return self.__align(self.continuous.resample(n).transpose())
    
    def __sample_sklearn_kde(self, n, seed = None) :
        return self.__align(self.continuous.sample(n_samples = n, random_state = seed))
    
    def __sample_scipy_rv_c(self, n, seed = None) :
        return self.__align(self.continuous.rvs(size = n, random_state = seed))
    
    
    def __get_len(self, indices): 
        if len(indices[0])==0: 
            return max(self.indices[1])+1
                       
        if len(indices[1])==0: 
            return max(self.indices[0])+1
        
        return max(max(self.indices[0]),max(self.indices[1])) +1
    
    def __check_indices(self, uncertain_obj, certain_data, indices):
        assert type(indices) == list
        assert len(indices) == 2
        
        
        comp_indices = [*indices[0], *indices[1]]
        for i in range(self.__get_len(indices)):
            assert i in comp_indices
            
            
        if len(indices[1]) == 0:
            assert certain_data == None
        
        if certain_data is None: 
            assert len(indices[1]) == 0 
        
        if len(indices[0]) == 0 and uncertain_obj == None:
            return True
        
        if type(uncertain_obj) == scipy.stats._kde.gaussian_kde:
            return len(indices[0]) == uncertain_obj.dataset.shape[0]
        
        if type (uncertain_obj) == sklearn.neighbors._kde.KernelDensity:
            return len(indices[0]) == uncertain_obj.n_features_in_

        if (issubclass(type(uncertain_obj), scipy.stats.rv_continuous) or 
            issubclass(type(uncertain_obj), scipy.stats._distn_infrastructure.rv_continuous_frozen)):
            return len(indices[0]) == 1
            
        return True
    
    def __align(self,uncertain_values): 

        ret = np.zeros((uncertain_values.shape[0], self.n_vars))
        ret[:,self.indices[0]] = np.array(uncertain_values)
        ret[:,self.indices[1]] = self.certain_data
        return ret
    
