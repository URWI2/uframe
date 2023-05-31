import numpy as np 
from uframe_instance import uframe_instance
import scipy
from scipy import stats 

#make only a private attribute _col_dtype, which is settled automatically
#check for contradictions with categorical variables, but only add it after we have categorical variables 
#not needed for noisy array function 

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
        self._columns=[]
        self._colnames={}
        self.rows=[]
        self._rownames={}
        self.col_dtype=[]
        self._col_dtype={}
        
        return 
    
    #incomplete 
    def append(self, new=None, colnames=None, rownames=None):
        
        print('pending, need to evaluate what data is given to the function')
        
        if type(new)==np.ndarray:
            self.append_from_numpy(new)
        
        elif type(new)==scipy.stats._kde.gaussian_kde:
            self.append_from_scipy_kde([new])
        
        elif type(new)== uframe_instance:
            self.append_from_uframe_instance([new])
        
        #append rownames to data 
        if rownames is not None:
            #check rownames for duplicates (also with previously existing rownames)
            if len(set(self.rows + rownames))!= len(self.rows) + len(rownames):
                raise ValueError("Duplicates among rownames")
            if len(rownames)!= len(new):
                raise ValueError("Number of rownames given does not match number of rows given")
            
            self.rows = self.rows + rownames
            self._rownames.update({rownames[i]: len(self.data)-len(new)+i for i in range(len(rownames))})
        else:
            
            self.rows = self.rows + [*list(range(len(self.data)-len(new), len(self.data)))]
            self._rownames.update({i:i for i in range(len(self.data)-len(new), len(self.data))})
        
        return 
    
    #append a numpy array with certain data (2D-array)
    #have yet to treat the col_dtype aspect in all append functions 
    def append_from_numpy(self, new=None, colnames=None):
        
        if len(self.columns)>0:
            if new.shape[1]!= len(self.columns):
                raise ValueError("New data does not match shape of uframe")
               
            else:
                for i in range(new.shape[0]):
                    self.data.append(uframe_instance(uncertain_obj=None, certain_data=new[i,:], 
                                                     indices=[[],[*list(range(new.shape[1]))]]))
        else:
            for i in range(new.shape[0]):
                self.data.append(uframe_instance(uncertain_obj=None, certain_data=new[i,:], 
                                                 indices=[[],[*list(range(new.shape[1]))]]))
                
            if colnames is None:
                self._columns= [*list(range(new.shape[1]))]
                self._colnames = {i:i for i in range(0, new.shape[1]) }
            else:
                if len(colnames)!=new.shape[1]:
                    raise ValueError("Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name:i for i, name in enumerate(colnames)}
                           
        return 
    
    #brauche analoge Funktion for sklearn kdes 
    #das müsste angepasst werden (bei sklearn: n_features_in)
    #muss rownames noch beachten 
    #Attribut new ist kernel_list
    def append_from_scipy_kde(self, kernel_list, colnames=None):
        
        if len(self.columns)>0:
            
            dimensions = len([kernel.d for kernel in kernel_list if kernel.d==len(self.columns)])
            if dimensions!=len(self.columns):
                raise ValueError("Dimension of list element does not match dimension of uframe")
         
        else:
            #check if all kernels have the same dimension 
            dimensions = len([kernel.d for kernel in kernel_list if kernel.d==kernel_list[0].d])
            if dimensions!=len(kernel_list):
                raise ValueError("Kernels in list must have same dimension")
        
            if colnames is None:
                self._columns = [*list(range(kernel_list[0].d))]
                self._colnames = {i:i for i in range(kernel_list[0].d)}
            else:
                if len(colnames)!=kernel_list[0].d:
                    raise ValueError("Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name:i for i, name in enumerate(colnames)}
                
        for i, kernel in enumerate(kernel_list):
            self.data.append(uframe_instance(uncertain_obj=kernel, certain_data=None,
                                             indices=[[*list(range(kernel.d))],[]]))
        
        return 
    
    #assume dictionary with indices of incomplete lines as keys and scipy kdes as values 
    #nan values for uncertain values in array certain 
    #have to add colnames (and possibly rownames as parameters instead of the defaults)
    def append_from_mix_with_distr(self, certain, distr, colnames=None):

        assert len(certain)==len(distr)
        
        if len(self.columns)>0:
            
            if len(self.columns)!=certain.shape[1]:
                raise ValueError("Dimensions of new data do not match uframe")
            
        else:
            if colnames is None:
                self._columns = [*list(range(certain.shape[1]))]
                self._colnames = {i:i for i in range(certain.shape[1])}
            else:
                if len(colnames)!=certain.shape[1]:
                    raise ValueError("Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name:i for i, name in enumerate(colnames)}
        
        for i in range(len(certain)):
            
            if i in distr.keys():
                assert len(list(np.where(np.isnan(certain[i]==True))))+ distr[i].d == len(self.columns)
            else:
                assert len(list(np.where(np.isnan(certain[i]==True)))) == len(self.columns)
          
            self.data.append(uframe_instance(uncertain_obj=distr[i], 
                                             certain_data=certain[i][np.isnan(certain[i])==False], 
                                             indices = [list(np.where(np.isnan(certain[i]==True))),
                                                        list(np.where(np.isnan(certain[i]==False)))]))
        return 
    
    #append from a list of uframe_instance objects
    #treat case of a single uframe_instance object in the append function 
    def append_from_uframe_instance(self, instances, colnames=None):
        
        #check that instances is really a list of uframe objects and has right length
        #do that in the append function 
        if len(instances)<1:
            return 
        
        if len(self.columns)>0:
            
            dimensions = len([instance.n_vars for instance in instances if instance.n_vars==len(self.columns)])
            if dimensions!=len(instances):
                raise ValueError("Dimensions of new instances do not match dimension of uframe")
        
        #treat colnames parameter here in else case
        else:
            if colnames is None:
                self._columns = [*list(range(instances[0].n_vars))]
                self._colnames = {i:i for i in range(instances[0].n_vars)}
            else:
                if len(colnames)!=instances[0].n_vars:
                    raise ValueError("Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name:i for i, name in enumerate(colnames)}
                
        
        self.data = self.data + instances
        
        return 
    
    #use scipy kde for Gaussian kernel and sklearn kde otherwise 
    #samples: list of length (n_instances) of np.ndarray of shape (dim_samples, n_samples) oder listen von np.arrays (dim_samples,)
    #erstellt kernels und greift auf bereits existierende append-Funktion zurück
    #bislang nur implementiert für scipy Gaussian kernel 
    #andere Kernels, welche auf sklearn basieren, folgen 
    def append_from_samples(self, samples_list, kernel='Gaussian', colnames=None):
        
        if len(samples_list)<1:
            raise ValueError("No samples given")
        
        kernel_list=[]
        
        for i, samples in enumerate(samples_list):
            
            if type(samples)==list:
                samples = np.array(samples).T
            
            #dim = samples.shape[0]
            
            if kernel=='Gaussian':
                kernel = stats.gaussian_kde(values)
            else:
                raise NotImplementedError()
            
            kernel_list.append(kernel)
        
        if kernel=='Gaussian':
            self.append_from_scipy_kde(kernel_list, colnames=colnames)
        
        return     
        
    def __repr__(self): 

        return "Object of class uframe"
    
        
    def __str__(self):
        
        return 'pending'
    
    #a function where a numpy array with nan for uncertain values is returned 

    #append from samples 
    #Funktion, welche aus np array durch Löschen und Imputation uframe macht 
    
    #also missing: functions for renaming or reordering columns/ rows 
    #need consensus how the renaming and reordering of columns/ rows should work 
    
    @property
    def columns(self):
        return self._columns 
    
    @columns.setter
    def columns(self, new_columns):
        
        if len(self.columns)>0 and len(new_columns)!=self._columns:
            raise ValueError("Length of new column list does not match column number of uframe")
        
        if len(set(new_columns))!=len(new_columns):
            raise ValueError("New column list contains duplicates!")
            
        #update _colnames dictionary if it is not empty

        for i, new_name in enumerate(new_columns):
            
            self._colnames[new_name]= self._colnames.pop(self.columns[i])
    
        self._columns = new_columns
        return 
    

    #function for reordering columns and rows
    def reorder_columns(self):
        raise NotImplementedError 
    
    
    def modal(self):
        
        return np.squeeze(np.array([inst.modal() for inst in self.data]), axis=0)
 
    #does not work for n>1 yet, see uframe_instance class
    #may have to change that later
    
    def sample(self, n=1, seed = None): 
        
        return np.squeeze(np.array([inst.sample(n, seed) for inst in self.data]), axis=0)
        
    def ev(self): 
        print('pending')
        return 
    
    def get_dummies(self):
        print('pending')
        return 
    
    def __getitem__ (self, index): 
        print('pending')
        return self.data[index]

#TO DO: MORE APPEND FUNCTIONS; INCLUDE THEM ALL IN THE APPEND FUNCTION
#COLNAMES AND ROWNAMES HANDLING (ALSO AT APPEND)
#ALLOW COLNAMES SETTING IN APPEND ONLY IF THERE ARE NO ELEMENTS YET; OTHERWISE COLNAMES MUST BE CHANGED SEPARATELY7
#WHAT TO DO ABOUT COL_DTYPE? DISCUSS THAT

#takes np array, randomly picks percentage of values unc_percentage and introduces uncertainty there
#default for imp_technique: Gaussian, which just adds a Gaussian noise to the attributes
def uframe_from_array(a:np.ndarray, imp_technique = 'mice', unc_percentage=0.1):
    return 

#here: add Gaussian noise of given std to chosen entries 
#relative=True: multiply std with standard deviation of the column to get the std for a column 
def uframe_noisy_array(a:np.ndarray, std=0.1, relative=False, unc_percentage = 0.1):
    return 

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
    print(b.sample(n=1))
    print(b.modal())
    b.append(new=np.identity(2))
    uframe_i = b.data[1]
    print(type(uframe_i))
    b.append_from_uframe_instance([uframe_i])
        