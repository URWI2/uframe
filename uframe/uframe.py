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
    
    def append(self):
        print('pending, need to evaluate what data is given to the function')
        
    
    #append a numpy array with certain data (2D-array)
    def append_from_numpy(self, new=None, rownames= None):
        
        if len(self.columns)>0:
            if new.shape[1]!= len(self.columns):
                print("New data does not match shape of uframe")
                return 
            else:
                for i in range(new.shape[0]):
                    self.data.append(uframe_instance(uncertain_obj=None, certain_data=new[i,:], 
                                                     indices=[[],list(range(new.shape[1]))]))
        else:
            for i in range(new.shape[0]):
                self.data.append(uframe_instance(uncertain_obj=None, certain_data=new[i,:], 
                                                 indices=[[],list(range(new.shape[1]))]))
            self.columns= list(range(new.shape[1]))
            self._colnames = {i:i for i in range(0, new.shape[1]) }
            
        #append the rownames of the new data, this should not depend on data type 
        
        self.rows.append(rownames)
        self._rownames.append({rownames[i]: len(self.data)-new.shape[0]+i for i in range(len(rownames))})
        
        return 
        
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

if __name__=="__main__":
    a = uframe()
    a.append_from_numpy(new=np.identity(3))
        