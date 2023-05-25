"""
uframe_instance class used in uframe to depict a single uncertain data instance.

"""
import scipy 

class uframe_instance(): 
    """
   A class used to represent an singel uncertain data instance.

   ...

   Attributes
   ----------
   continuous : class
       A class describing the underlying uncertainty.
   certain : list[float] 
       List of certain values
   colnames : list[str]
       associations from colnames and indices to dimensions of continuous and certain

   Methods
   -------
   sample(n = 1, seed = None)
       Samples n samples from the uncertain instance
       
   """
   
    def __init__(self,data): 
        
        if type(data) == scipy.stats._kde.gaussian_kde:
            self.__init_scipy_kde(data)
            
    
    def sample(self,n: int = 1, seed: int= None): 
        
        if type(self.continuous) == scipy.stats._kde.gaussian_kde:
            return self.__sample_scipy_kde(n)
        
    
    def __str__(self):
        print("Data Instance")
        
    def __repr__(self):
        print("Data Instance")
        
        
    def __init_scipy_kde(self, kernel):
        self.continuous = kernel
    
    def __sample_scipy_kde(self, n): 
        return self.continuous.resample(n)
