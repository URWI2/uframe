"""
uframe_instance class used in uframe to depict a single uncertain data instance.

"""
import scipy 
import numpy as np

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
            assert  type(uncertain_obj) in [scipy.stats._kde.gaussian_kde]
        if certain_data is not None:
            assert type(certain_data) in [np.ndarray,np.array]
            assert len(certain_data) == len(indices[1])
            
        assert type(indices) == list
        assert len(indices) == 2
        
        
        
        
        if type(uncertain_obj) == scipy.stats._kde.gaussian_kde:
            self.__init_scipy_kde(uncertain_obj)
       
        self.certain_data = certain_data 
        self.indices = indices
        self.n_vars = self.__get_len(indices)
        
    def sample(self,n: int = 1, seed: int= None): 
        
        if type(self.continuous) == scipy.stats._kde.gaussian_kde:
            return self.__align(self.__sample_scipy_kde(n))
        
    def modal(self): 
        
        if type(self.continuous) == scipy.stats._kde.gaussian_kde:
            return self.__align(self.__modal_scipy_kde())
        
        
        
    def __str__(self):
        print("Data Instance")
        
    def __repr__(self):
        print("Data Instance")
    
    def __len__(self): 
        return self.n_vars
        
    def __modal_scipy_kde(self): 
        opt = scipy.optimize.basinhopping(lambda x: -self.continuous.pdf(x),[0,0] )
        return opt.x
        
    def __init_scipy_kde(self, kernel):
        self.continuous = kernel
    
    def __sample_scipy_kde(self, n): 
        return self.continuous.resample(n).transpose()
    def __get_len(self, indices): 
        if len(indices[0])==0: 
            return max(self.indices[1])+1
                       
        if len(indices[1])==0: 
            return max(self.indices[0])+1
        
        return max(max(self.indices[0]),max(self.indices[0])) +1
    
    
    def __align(self,uncertain_values): 
        
        ret = np.zeros(self.n_vars)
        ret[self.indices[0]] = np.array(uncertain_values)
        ret[self.indices[1]] = self.certain_data
        return ret
    
