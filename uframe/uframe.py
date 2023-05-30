import numpy as np 
from uframe_instance import uframe_instance
import scipy

class uframe(): 
    """
    A class used for storing and working with uncertain data. 

    ...

    ----------
    data : list
        A list of data instances of class uframe_instance
    columns: list
        List of column names
    rows: list
        List of row names
    _colnames: dict
        A dictionary with column names as keys and the corresponding indices in 
        the data as values
    _rownames: dict
        A dictionary with row names as keys and the corresponding indices in 
        the data as values
    col_dtype: list
        A list of the datatypes of the columns
    _col_dtype: dict
        A dictionary with column names as keys and the corresponding data 
        types as values 
   
    Methods
    -------
    modal()
        Returns a numpy array of the data, where all uncertain values are 
        replaced with their modal values.
    
    sample(n = 1, seed = None)
        Samples n samples for each instance and returns one numpy array, 
        where n samples from each instance are below each other.
   
    ev()
        Returns a numpy array where all uncertain values are replaced with 
        their expected values
    
    get_dummies()
        Performs one hot encoding on all categorical columns in the data.
        Have yet to determine based on what a column is identified as 
        categorical
    
    update():
        Update specific instance, not immediately relevant
    
    append(new = None):
        Append new rows to the data which can be either certain or uncertain.
        In place. New rows have to be numpy arrays at first, later add
        different sources of uncertain new rows. 
        
        
    """
    def __init__(self): 
        
        self.data= []
        self.columns=[]
        self.rows=[]
        self._colnames={}
        self._rownames={}
        self.col_dtype=[]
        self._col_dtype={}
        
        return 
    
    def append(self, new=None):
        print('pending, need to evaluate what data is given to the function')
        
        if type(new)==np.ndarray:
            self.append_from_numpy(self.new)
        
        return 
    
    #append a numpy array with certain data (2D-array)
    def append_from_numpy(self, new=None, rownames= None):
        
        if len(self.columns)>0:
            if new.shape[1]!= len(self.columns):
                print("New data does not match shape of uframe")
                return 
            else:
                for i in range(new.shape[0]):
                    self.data.append(uframe_instance(uncertain_obj=None, certain_data=new[i,:], 
                                                     indices=[[],[*list(range(new.shape[1]))]]))
        else:
            for i in range(new.shape[0]):
                self.data.append(uframe_instance(uncertain_obj=None, certain_data=new[i,:], 
                                                 indices=[[],[*list(range(new.shape[1]))]]))
            self.columns= [*list(range(new.shape[1]))]
            self._colnames = {i:i for i in range(0, new.shape[1]) }
            
        #append the rownames of the new data, this should not depend on data type 
        
        if rownames!=None:
            self.rows = self.rows + rownames
            self._rownames.update({rownames[i]: len(self.data)-new.shape[0]+i for i in range(len(rownames))})
        else:
            self.rows = self.rows + [*list(range(len(self.data)-new.shape[0], len(self.data)))]
            self._rownames.update({i:i for i in range(len(self.data)-new.shape[0], len(self.data))})
        
        return 
    
    #missing: more append functions (from scipy.stats.gaussian_kde next (resp. list of those))
    #append from list of distributions
    #append from uframe instances
    #append from yet to be defined mixtures of certain values and distributions
    #for example the np.array with None values and corresponding dict of multidimensional kdes 
    #(see Felde bachelor thesis for example)
    
    def append_from_scipy_kde(self, new):
        return 
    
    
    
    def __repr__(self): 

        return "Object of class uframe"
    
        
    def __str__(self):
        
        return 'pending'

    #needs to be tested with an uncertain instance, need to fix modal so that it works
    #if there are no uncertain variables in the uframe_instance
    
    #also missing: functions for renaming or reordering columns/ rows 
    #need decorators for that
    
    def modal(self):
        
        res = np.array([inst.modal() for inst in self.data])

        return res
    
    #also untested as of now, need proper test instance and test script 
    def sample(self, n=1, seed = None): 
        
        res = np.array([inst.sample(n, seed) for inst in self.data])
        
        return res 

    def ev(self): 
        print('pending')
        return 
        
    def get_dummies(self):
        print('pending')
        return 
    
    def __getitem__ (self, index): 
        print('pending')
        return self.data[index]


if __name__=="__main__":
    a = uframe()
    a.append_from_numpy(new=np.identity(3))
        