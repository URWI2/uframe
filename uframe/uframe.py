import numpy as np 
from uframe_instance import uframe_instance
import scipy
from scipy import stats 

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
    
    def append(self, new=None, rownames=None):
        print('pending, need to evaluate what data is given to the function')
        
        if rownames!=None:
            assert len(new)==len(rownames)
        
        if type(new)==np.ndarray:
            self.append_from_numpy(new, rownames)
        
        elif type(new)==scipy.stats._kde.gaussian_kde:
            self.append_from_scipy_kde([new], rownames)
        
        elif type(new)== uframe_instance:
            self.append_from_uframe_instance([new])
        
        
        return 
    
    #append a numpy array with certain data (2D-array)
    #have yet to treat the col_dtype aspect in both cases
    #in this case, have float columns (what about the integer case?)
    #do we allow a coltypes parameter for this function? have to check if it is
    #a valid list of coltypes then that matches the array
    
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
            self._rownames.update({rownames[i]: len(self.data)-len(new)+i for i in range(len(rownames))})
        else:
            self.rows = self.rows + [*list(range(len(self.data)-len(new), len(self.data)))]
            self._rownames.update({i:i for i in range(len(self.data)-len(new), len(self.data))})
        
        return 
    
    #missing: more append functions (from scipy.stats.gaussian_kde next (resp. list of those))
    #append from list of distributions
    #append from uframe instances
    #append from yet to be defined mixtures of certain values and distributions

    
    def append_from_scipy_kde(self, kernel_list, rownames=None):
        
        if len(self.columns)>0:
            
            dimensions = len([kernel.d for kernel in kernel_list if kernel.d==len(self.columns)])
            if dimensions!=len(self.columns):
                print("Dimension of list element does not match dimension of uframe")
                return 
        else:
            #check if all kernels have the same dimension 
            dimensions = len([kernel.d for kernel in kernel_list if kernel.d==kernel_list[0].d])
            if dimensions!=len(kernel_list):
                print("Kernels in list must have same dimension")
                return 
            self.columns = [*list(range(kernel_list[0].d))]
            self._colnames = {i:i for i in range(kernel_list[0].d)}
        
        for i, kernel in enumerate(kernel_list):
            self.data.append(uframe_instance(uncertain_obj=kernel, certain_data=None,
                                             indices=[[*list(range(kernel.d))],[]]))
        
        #treat rownames and data types for later, maybe do that in the general append function anyway 
        return 
    
    #assume dictionary with indices of incomplete lines as keys and scipy kdes as values
    #nan values for uncertain values in array certain 
    #have to add colnames (and possibly rownames as parameters instead of the defaults)
    def append_from_mix_with_distr(self, certain, distr):

        assert len(certain)==len(distr)
        
        if len(self.columns)>0:
            
            if len(self.columns)!=certain.shape[1]:
                print("Dimensions of new data do not match uframe")
                return 
            
            #check if kernel dimension matches None values per row, what is best option if not?
            
        else:
            self.columns = [*list(range(certain.shape[1]))]
            #das sollte doch später automatisch passieren, falls decorator geeignet geschrieben, richtig?
            self._colnames = {i:i for i in range(certain.shape[1])}
        
        for i in range(len(certain)):
            self.data.append(uframe_instance(uncertain_obj=distr[i], 
                                             certain_data=certain[i][np.isnan(certain[i])==False], 
                                             indices = [list(np.where(np.isnan(certain[i]==True))),
                                                        list(np.where(np.isnan(certain[i]==False)))]))
        
        return 
    
    #append from a list of uframe_instance objects
    #treat case of a single uframe_instance object in the append function 
    def append_from_uframe_instance(self, instances):
        
        if len(self.columns)>0:
            
            dimensions = len([instance.n_vars for instance in instances if instance.n_vars==len(self.columns)])
            if dimensions!=len(instances):
                print("Dimensions of new instances do not match dimension of uframe")
                return 
        
        self.data = self.data + instances
        
        return 
                
        
    def __repr__(self): 

        return "Object of class uframe"
    
        
    def __str__(self):
        
        return 'pending'

    #needs to be tested with an uncertain instance, need to fix modal so that it works
    #if there are no uncertain variables in the uframe_instance
    
    #also missing: functions for renaming or reordering columns/ rows 
    #need decorators for that
    
    # #setter function for columns
    # @columns.setter
    # def columns(new_columns):
        
    #     #check if column names are unique
    #     #check if the length of the new columns is right (allow dict with new names)
    #     #update underlying _colnames dict 
    #     return 
    
    def modal(self):
        
        res = np.array([inst.modal() for inst in self.data])

        return res
    
    #does not work for n>1 yet 
    def sample(self, n=1, seed = None): 
        
        res = [inst.sample(n, seed) for inst in self.data]
        
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
    
    def measure(n):
        m1 = np.random.normal(size=n)
        m2 = np.random.normal(scale=0.5, size=n)
        return m1+m2, m1-m2

    m1, m2 = measure(2000)
    m1, m2 = measure(2000)
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    
    kernel_list = [kernel]
    
    b = uframe()
    b.append_from_scipy_kde(kernel_list)
    b.append(new=np.identity(2))
    uframe_i = b.data[1]
    print(type(uframe_i))
    b.append_from_uframe_instance([uframe_i])
        