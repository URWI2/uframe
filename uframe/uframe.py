import numpy as np 
from uframe_instance import uframe_instance
import scipy
from scipy import stats 
from sklearn.neighbors import KernelDensity
import sklearn.neighbors 

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
        self._rows=[]
        self._rownames={}
        self.col_dtype=[]
        self._col_dtype={}
        
        return 
    
    #incomplete, does not work yet
    #parameter for whether new consists of sampled data 
    #falls sampled==True, wird new als Liste/ Array von Samples erwartet und versucht, append_from_samples zu callen
    def append(self, new=None, colnames=None, rownames=None):
        
        
        if type(new)==np.ndarray:
            self.append_from_numpy(new, colnames)
        
        elif type(new)==scipy.stats._kde.gaussian_kde:
            self.append_from_scipy_kde([new], colnames)
            
        elif type(new)==sklearn.neighbors._kde.KernelDensity:
            self.append_from_sklearn_kde([new], colnames)
        
        elif type(new)== uframe_instance:
            self.append_from_uframe_instance([new], colnames)
        
        elif type(new)==list:
            
            print("Enter else case")
            
            if len(new)==2 and type(list[0])==np.ndarray and type(list[1])==dict:
                
                self.append_from_mix_with_distr(list[0], list[1], colnames)
            
            if type(new[0])==scipy.stats._kde.gaussian_kde:          
                self.append_from_scipy_kde(new, colnames)
            elif type(new[0])==sklearn.neighbors._kde.KernelDensity:
                self.append_from_sklearn_kde(new, colnames)
            elif type(new[0])==uframe_instance:
                self.append_from_uframe_instance(new, colnames)
        
        self.addRownames(new, rownames)
       
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
                self.data.append(uframe_instance(uncertain_obj=None, certain_data=new[i,:], indices=[[],[*list(range(new.shape[1]))]]))
                
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
           if dimensions!=len(kernel_list):
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
    
    def append_from_sklearn_kde(self, kernel_list, colnames=None):
        
        if len(self.columns)>0:
            
            dimensions = len([kernel.n_features_in_ for kernel in kernel_list if kernel.n_features_in_==len(self.columns)])
            if dimensions!=len(kernel_list):
                raise ValueError("Dimension of list element does not match dimension of uframe")
         
        else:
            #check if all kernels have the same dimension 
            dimensions = len([kernel.n_features_in_ for kernel in kernel_list if kernel.n_features_in_==kernel_list[0].d])
            if dimensions!=len(kernel_list):
                raise ValueError("Kernels in list must have same dimension")
        
            if colnames is None:
                self._columns = [*list(range(kernel_list[0].n_features_in_))]
                self._colnames = {i:i for i in range(kernel_list[0].n_features_in_)}
            else:
                if len(colnames)!=kernel_list[0].n_features_in_:
                    raise ValueError("Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name:i for i, name in enumerate(colnames)}
                
        for i, kernel in enumerate(kernel_list):
            self.data.append(uframe_instance(uncertain_obj=kernel, certain_data=None,
                                             indices=[[*list(range(kernel.n_features_in_))],[]]))
        
    #jeder Eintrag von distr_list muss entweder eine multivariate Verteilung über alle Variablen
    #oder eine Liste unabhängiger Verteilungen, deren Dimensionen aufsummiert die Gesamtdimension ergeben, sein   
    #ebenfalls noch nicht getestet 
    def append_from_rv_cont(self, distr_list, colnames=None):
        
        dimensions_mv = [distr.rvs(size=1).shape[0] for distr in distr_list if type(distr)!=list]
        dimensions_dlists = [sum([d.rvs(size=1).shape[0] for d in distr]) for distr in distr_list if type(distr)==list]
        d_list = dimensions_mv + dimensions_dlists
        if len(self.columns)>0:
            matches = len([i for i in d_list if i==len(self.columns)])
            if matches!=len(distr_list):
                raise ValueError("Dimension of distributions must match uframe dimension")
        else:
            matches = len([i for i in d_list if i==d_list[0]])
            if matches!=len(distr_list):
                raise ValueError("Distributions in list must have same dimension")
        
            if colnames is None:
                self._columns = [*list(range(d_list[0]))]
                self._colnames = {i:i for i in range(d_list[0])}
            else:
                if len(colnames)!= d_list[0]:
                    raise ValueError("Length of column list does not match dimension of distributions")
                else:
                    self._columns = colnames 
                    self._colnames = {name:i for i,name in enumerate(colnames)}
                    
        for i, distr in enumerate(distr_list):
            self.data.append(uframe_instance(uncertain_obj=distr, certain_data=None,
                                             indices=[[*list(range(d_list[0]))],[]]))
                
        return 
    
    #assume dictionary with indices of incomplete lines as keys and scipy kdes as values 
    #nan values for uncertain values in array certain 
    #have to add colnames (and possibly rownames as parameters instead of the defaults)
    #not yet tested 
    
    #should work for both scipy and sklearn kernels 
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
                if type(distr[i])==scipy.stats._kde.gaussian_kde:
                    assert len(list(np.where(np.isnan(certain[i]==True))))+ distr[i].d == len(self.columns)
                elif type(distr[i])==sklearn.neighbors._kde.KernelDensity:
                    assert len(list(np.where(np.isnan(certain[i]==True)))) + distr[i].n_features_in_ == len(self.columns)
            else:
                assert len(list(np.where(np.isnan(certain[i]==True)))) == len(self.columns)
          
            self.data.append(uframe_instance(uncertain_obj=distr[i], 
                                             certain_data=certain[i][np.isnan(certain[i])==False], 
                                             indices = [list(np.where(np.isnan(certain[i]==True))),
                                                        list(np.where(np.isnan(certain[i]==False)))]))
        return 
    
    
    def append_from_mix_with_rv_cont(self, certain, distr, colnames=None):
        return 
    
     
    #append from a list of uframe_instance objects
    def append_from_uframe_instance(self, instances, colnames=None):
        
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
    
    #samples: list of length (n_instances) of np.ndarray of shape (dim_samples, n_samples) oder listen von np.arrays (dim_samples,)
    #erstellt kernels und greift auf bereits existierende append-Funktion zurück
    #für scipy Gaussian und alle sklearn Kernels 
    def append_from_samples(self, samples_list, kernel='stats.gaussian_kde', colnames=None, rownames=None):
        
        if len(samples_list)<1:
            raise ValueError("No samples given")
        
        kernel_list=[]
        
        for i, samples in enumerate(samples_list):
            
            if type(samples)==list:
                samples = np.array(samples).T
               
            if kernel=='stats.gaussian_kde':
                kde = stats.gaussian_kde(values)
            elif kernel in ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']:
                samples = samples.T
                kde = KernelDensity(kernel=kernel, bandwidth=1.0).fit(samples)
            else:
                raise NotImplementedError("Given kernel does not exist")
            kernel_list.append(kde)
        
        if kernel=='stats.gaussian_kde':
            self.append_from_scipy_kde(kernel_list, colnames=colnames)
        else:
            self.append_from_sklearn_kde(kernel_list, colnames=colnames)
        
        self.addRownames(kernel_list, rownames)
        return  
    
#MISSING: APPEND FROM LISTS OF DISTRIBUTIONS OR MIXES WHERE THE CONTINUOUS ELEMENT IS A SCIPY DISTRIBUTION (or list of distributions)



    def addRownames(self, new, rownames):
       
        if rownames is not None:
            
            #check rownames for duplicates (also with previously existing rownames)
            if len(set(self._rows + rownames))!= len(self.rows) + len(rownames):
                raise ValueError("Duplicates among rownames")
            if len(rownames)!= len(new):
                raise ValueError("Number of rownames given does not match number of rows given")
            
            self._rows = self._rows + rownames
            self._rownames.update({rownames[i]: len(self.data)-len(new)+i for i in range(len(rownames))})
        else:
            
            self._rows = self.rows + [*list(range(len(self.data)-len(new), len(self.data)))]
            self._rownames.update({i:i for i in range(len(self.data)-len(new), len(self.data))})
        
        return 
    

    
    def __repr__(self): 
        
        return "Object of class uframe"
           
    def __str__(self):
        
        print("Object of class uframe with certain values:")
        print(self.array_rep())
        return ""
    
    #returns a np.array with the certain values and nan for uncertain values 
    #TO DO: COLUMNS AND ROWS GEMÄß DER EINTRÄGE IN ROWS UND COLUMNS AUSGEBEN; NICHT IN GESPEICHERTER FORM 
    def array_rep(self):
        
        x = np.zeros((len(self.data), len(self.columns)), dtype=np.float64)
        for i, instance in enumerate(self.data):
            x[i, instance.indices[1]]= instance.certain_data
            x[i, instance.indices[0]]= np.nan
        
        return x
    
    #also missing: functions for reordering columns/ rows 
    #renaming rows is also missing 
    #need consensus how the renaming and reordering of columns/ rows should work 
    
    @property
    def columns(self):
        return self._columns 
    
    @columns.setter
    def columns(self, new_columns):
        
        if len(self.columns)>0 and len(new_columns)!=len(self._columns):
            raise ValueError("Length of new column list does not match column number of uframe")
        
        if len(set(new_columns))!=len(new_columns):
            raise ValueError("New column list contains duplicates!")
            
        #update _colnames dictionary if it is not empty

        new_d={}
        for i, new_name in enumerate(new_columns):
            
            new_d[new_name]= self._colnames.pop(self.columns[i])
    
        self._colnames = new_d 
        self._columns = new_columns
        return 
    
    @property
    def rows(self):
        return self._rows

    #rows always have to be integers so far, probably keep it that way like pandas 
    @rows.setter
    def rows(self, new_rows):
        
        if len(self.rows)>0 and len(new_rows)!=len(self._rows):
            raise ValueError("Length of new row list does not match row number of uframe")
        
        if len(set(new_rows))!=len(new_rows):
            raise ValueError("New row list contains duplicates")
        
        #check that all row indices are integers
        if len([index for index in new_rows if type(index)==int])!=len(new_rows):
            raise ValueError("Indices of rows have to be integers")
        
        new_d = {}
        for i, new_index in enumerate(new_rows):
            
            new_d[new_index]= self._rownames.pop(self.rows[i])
            #self._rownames[new_index]= self._rownames.pop
        
        self._rownames= new_d
        self._rows = new_rows
        
        return 
            
    #function for reordering columns and rows
    #Parameter: Liste mit Teilmenge der Spalten in einer bestimmten Reihenfolge, interpretiere das als
    #neue Reihenfolge dieser Spalten, i.e., wenn bei Spalten [1,2,4] der Parameter [4,2,1] übergeben wird,
    #werden diese 3 Spalten so neu angeordnet, der Rest bleibt unverändert
    #dies wird nur für die Ausgabe in columns gespeichert (passe dann array_rep Funktion entsprechend an)
    #Frage: wird dann auf Input für append Funktionen entsprechend eingegangen und gemäß
    #_colnames vertauscht vor dem Speichern? Müsste aus Gründen der Benutzerfreundlichkeit eigentlich so passieren
    #würde in den append-Funktionen einem geeigneten Umsortieren gemäß self._colnames entsprechen 
    #noch zu besprechen 
    def reorder_columns(self):
        
        raise NotImplementedError() 
    
    #Shape-Problem: bei festen Werten wird array unterschiedlicher Shape zurückgegeben als mit unsicheren
    #uframe instance Problem 
    #nicht array aus Liste machen, sondern arrays in Liste concatenaten!
    #geht erst, wenn die zurückgegebenen array shapes einheitlich sind 
    def mode(self):
        
        return [inst.mode() for inst in self.data]
 
    #füge später noch den seed hinzu 
    #np.concatenate mit dieser Liste machen, sobald array Dimensionen passen 
    def sample(self, n=1, seed = None): 
        
        return [inst.sample(n, seed) for inst in self.data]
        
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
#see function generate missing values to in untitled file to select uncertain values 
def uframe_from_array(a:np.ndarray, imp_technique = 'mice', unc_percentage=0.1):
    
    return 

#here: add Gaussian noise of given std to chosen entries 
#relative=True: multiply std with standard deviation of the column to get the std for a column 
def uframe_noisy_array(a:np.ndarray, std=0.1, relative=False, unc_percentage = 0.1):
    
    
    return 

if __name__=="__main__":
    a = uframe()
    #a.append_from_numpy(new=np.identity(3))
    
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
    b.append(kernel_list)
    print(b.sample(n=8))
    print(b.mode())
    b.append(new=np.identity(2))
    uframe_i = b.data[1]
    print(type(uframe_i))
    b.append([uframe_i])
    
    b.append_from_samples([values, values, values], kernel='tophat')
    print(len(b.data))
    b.append(new=kernel_list)
    b.append([uframe_i])
    
    print(len(b.data))
