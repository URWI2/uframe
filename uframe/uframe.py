import numpy as np 
from uframe_instance import uframe_instance


class uframe(): 
    """
    A class used for storing and working with uncertain data. 

    
    ...

    Attributes
    ----------
    data : list
        A list of data instances of class uframe_instance
    certain_data : np.array 
        Numpy array of certain values
    indices : [list,list]
        associations of indices to continuous and certain

    Methods
    -------
    sample(n = 1, seed = None)
        Samples n samples for each instance and returns one numpy array, where n samples are bellow each other.
   
    ev()
        Returns a numpy array where all uncertain values are replaced with their expected values according to maximum likelihood.
    
    model()
        Returns a numpy array of the data, where all uncertain values are replaced with their modal values.
        
    
    
    """
    def __init__(self): 
        print('pending')
        
    def __repr__(self): 
        print('pending') 
    def __str__(self):
        print('pending')

    def modal(self):
        print('pending')
       
    def sample(self, n=1, seed = None): 
        print('pending')

    def ev(self): 
        print('pending')
        
    def get_dummies(self):
        print('pending')
        
    def __getitem__ (self, index): 
        print('pending')
        
        