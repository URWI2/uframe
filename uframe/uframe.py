import numpy as np
from uframe_instance import uframe_instance
import scipy
from scipy import stats
from sklearn.neighbors import KernelDensity
import sklearn.neighbors
import miceforest as mf 

# make only a private attribute _col_dtype, which is settled automatically
# check for contradictions with categorical variables, but only add it after we have categorical variables
# not needed for noisy array function


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

        self.data = []
        self._columns = []
        self._colnames = {}
        self._rows = []
        self._rownames = {}
        self.col_dtype = []
        self._col_dtype = {}

        return

    def append(self, new=None, colnames=None, rownames=None):
        
        if new is None:
            return 

        if type(new) == np.ndarray:
            self.append_from_numpy(new, colnames)

        elif type(new) == scipy.stats._kde.gaussian_kde:
            self.append_from_scipy_kde([new], colnames)

        elif type(new) == sklearn.neighbors._kde.KernelDensity:
            self.append_from_sklearn_kde([new], colnames)

        elif type(new) == uframe_instance:
            self.append_from_uframe_instance([new], colnames)
        
        elif issubclass(type(new), scipy.stats.rv_continuous) or issubclass(type(new), 
                                                        scipy.stats._distn_infrastructure.rv_continuous_frozen):
            self.append_from_rv_cont([new], colnames)
        
        elif issubclass(type(new), scipy.stats._multivariate.multi_rv_generic) or issubclass(type(new), 
                                                                    scipy.stats._multivariate.multi_rv_frozen):
            self.append_from_rv_cont([new], colnames)

        elif type(new) == list:

            if len(new) == 2 and type(new[0]) == np.ndarray and type(new[1]) == dict:

                
                self.append_from_mix_with_distr(new[0], new[1], colnames)

            #add a len(new)==3 case here where a dictionary of categorical variables is third
            #and a 3-part mixed append function is called 
            #also a case where only a list of dicts for cat. variables is appended
            #one with two lists (or a list of lists of length 2) with only continuous und categorical variables
            
            if type(new[0]) == scipy.stats._kde.gaussian_kde:
                self.append_from_scipy_kde(new, colnames)
            elif type(new[0]) == sklearn.neighbors._kde.KernelDensity:
                self.append_from_sklearn_kde(new, colnames)
            elif type(new[0]) == uframe_instance:
                self.append_from_uframe_instance(new, colnames)
            elif issubclass(type(new[0]), scipy.stats.rv_continuous) or issubclass(type(new[0]), scipy.stats._distn_infrastructure.rv_continuous_frozen):
                self.append_from_rv_cont(new, colnames)
            elif issubclass(type(new[0]), scipy.stats._multivariate.multi_rv_generic) or issubclass(type(new[0]), 
                                                                          scipy.stats._multivariate.multi_rv_frozen):
                self.append_from_rv_cont(new, colnames)
            #Fall, wo Liste von Listen von 1D Distribs übergeben wird     
            elif type(new[0])==list:
                self.append_from_rv_cont(new, colnames)
            
      
        if type(new) is not list:
            self.addRownames([new], rownames)
        elif type(new[0])==np.ndarray:
            self.addRownames(new[0], rownames)
        else:
            self.addRownames(new, rownames)

        return

    # append a numpy array with certain data (2D-array)
    # have yet to treat the col_dtype aspect in all append functions
    def append_from_numpy(self, new=None, colnames=None):

        if len(self.columns) > 0:
            if new.shape[1] != len(self.columns):
                raise ValueError("New data does not match shape of uframe")

            else:
                for i in range(new.shape[0]):
                    self.data.append(uframe_instance(uncertain_obj=None, certain_data=new[i, :],
                                                     indices=[[], [*list(range(new.shape[1]))]]))
        else:
            for i in range(new.shape[0]):
                self.data.append(uframe_instance(uncertain_obj=None, certain_data=new[i, :], indices=[
                                 [], [*list(range(new.shape[1]))]]))

            if colnames is None:
                self._columns = [*list(range(new.shape[1]))]
                self._colnames = {i: i for i in range(0, new.shape[1])}
            else:
                if len(colnames) != new.shape[1]:
                    raise ValueError(
                        "Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name: i for i,
                                      name in enumerate(colnames)}

        return

    def append_from_scipy_kde(self, kernel_list, colnames=None):

        if len(self.columns) > 0:

            dimensions = len(
                [kernel.d for kernel in kernel_list if kernel.d == len(self.columns)])
            if dimensions != len(kernel_list):
                raise ValueError(
                    "Dimension of list element does not match dimension of uframe")

        else:
            # check if all kernels have the same dimension
            dimensions = len(
                [kernel.d for kernel in kernel_list if kernel.d == kernel_list[0].d])
            if dimensions != len(kernel_list):
                raise ValueError("Kernels in list must have same dimension")

            if colnames is None:
                self._columns = [*list(range(kernel_list[0].d))]
                self._colnames = {i: i for i in range(kernel_list[0].d)}
            else:
                if len(colnames) != kernel_list[0].d:
                    raise ValueError(
                        "Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name: i for i,
                                      name in enumerate(colnames)}

        for i, kernel in enumerate(kernel_list):
            self.data.append(uframe_instance(uncertain_obj=kernel, certain_data=None,
                                             indices=[[*list(range(kernel.d))], []]))

        return

    def append_from_sklearn_kde(self, kernel_list, colnames=None):

        if len(self.columns) > 0:

            dimensions = len(
                [kernel.n_features_in_ for kernel in kernel_list if kernel.n_features_in_ == len(self.columns)])
            if dimensions != len(kernel_list):
                raise ValueError(
                    "Dimension of list element does not match dimension of uframe")

        else:
            # check if all kernels have the same dimension
            dimensions = len(
                [kernel.n_features_in_ for kernel in kernel_list if kernel.n_features_in_ == kernel_list[0].d])
            if dimensions != len(kernel_list):
                raise ValueError("Kernels in list must have same dimension")

            if colnames is None:
                self._columns = [*list(range(kernel_list[0].n_features_in_))]
                self._colnames = {i: i for i in range(
                    kernel_list[0].n_features_in_)}
            else:
                if len(colnames) != kernel_list[0].n_features_in_:
                    raise ValueError(
                        "Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name: i for i,
                                      name in enumerate(colnames)}

        for i, kernel in enumerate(kernel_list):
            self.data.append(uframe_instance(uncertain_obj=kernel, certain_data=None,
                                             indices=[[*list(range(kernel.n_features_in_))], []]))

    # jeder Eintrag von distr_list muss entweder eine multivariate Verteilung über alle Variablen
    # oder eine Liste unabhängiger 1D-Verteilungen
    def append_from_rv_cont(self, distr_list, colnames=None):

       
        dimensions_mv = [distr.mean.shape[0]
                         for distr in distr_list if issubclass(type(distr), scipy.stats._multivariate.multi_rv_generic) or
                         issubclass(type(distr), scipy.stats._multivariate.multi_rv_frozen)]
        
        dimensions_rv = [distr.rvs(size=1).shape[0] for distr in distr_list if 
                         issubclass(type(distr), scipy.stats._distn_infrastructure.rv_continuous_frozen) or
                         issubclass(type(distr), scipy.stats.rv_continuous)]
        
        dimensions_dlists = [sum([d.rvs(size=1).shape[0] for d in distr])
                             for distr in distr_list if type(distr) == list]
        
        d_list = dimensions_mv + dimensions_rv + dimensions_dlists
        if len(self.columns) > 0:
            matches = len([i for i in d_list if i == len(self.columns)])
            if matches != len(distr_list):
                raise ValueError(
                    "Dimension of distributions must match uframe dimension")
        else:
            matches = len([i for i in d_list if i == d_list[0]])
            if matches != len(distr_list):
                raise ValueError(
                    "Distributions in list must have same dimension")

            if colnames is None:
                self._columns = [*list(range(d_list[0]))]
                self._colnames = {i: i for i in range(d_list[0])}
            else:
                if len(colnames) != d_list[0]:
                    raise ValueError(
                        "Length of column list does not match dimension of distributions")
                else:
                    self._columns = colnames
                    self._colnames = {name: i for i,
                                      name in enumerate(colnames)}

        #every element of distr list is either a 1D RV, a MV Gaussian or a list of 1D RVs 
        for i, distr in enumerate(distr_list):
            
            if issubclass(type(distr), scipy.stats._distn_infrastructure.rv_continuous_frozen) or issubclass(type(distr), 
                                                                                        scipy.stats.rv_continuous):
                assert distr.rvs(size=1).shape[0]==len(self.columns)
            elif issubclass(type(distr), scipy.stats._multivariate.multi_rv_generic) or issubclass(type(distr), 
                                                                        scipy.stats._multivariate.multi_rv_frozen):
               
                assert distr.mean.shape[0]==len(self.columns)
            #checke, dass die Liste nur 1D-Verteilungen enthält
            #erlaube Liste mit multivariater Verteilung zunächst nicht, mit Christian absprechen, ob seine Klasse das kann 
            else:
                list_len = len([l for l in distr if issubclass(type(l), scipy.stats.rv_continuous)
                                or issubclass(type(l), scipy.stats._distn_infrastructure.rv_continuous_frozen)])
                assert list_len==len(self.columns)
                assert len(distr)==len(self.columns)
            self.data.append(uframe_instance(uncertain_obj=distr, certain_data=None,
                                             indices=[[*list(range(d_list[0]))], []]))

        return

    # assume dictionary with indices of incomplete lines as keys and scipy kdes as values
    # nan values for uncertain values in array certain
    #NOCH KANN KEIN LEERES ARRAY FÜR CERTAIN DATA ÜBERGEBEN WERDEN; NOCH ZU TUN FÜR CHRISTIAN
    def append_from_mix_with_distr(self, certain, distr, colnames=None):

        if len(self.columns) > 0:

            if len(self.columns) != certain.shape[1]:
                raise ValueError("Dimensions of new data do not match uframe")

        else:
            if colnames is None:
                self._columns = [*list(range(certain.shape[1]))]
                self._colnames = {i: i for i in range(certain.shape[1])}
            else:
                if len(colnames) != certain.shape[1]:
                    raise ValueError(
                        "Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name: i for i,
                                      name in enumerate(colnames)}

        for i in range(len(certain)):

            if i in distr.keys():

                # checke hier auch für Liste von 1D RVs (rv_continuous oder RV cont. frozen) oder multidim. RV
                # oder 1 fehlendes Attribut und 1D RV cont oder cont frozen
                if type(distr[i]) == scipy.stats._kde.gaussian_kde:
                    print(len(np.where(np.isnan(certain[i]) == False)[0]))
                    print(distr[i].d)
                    print("Columns", len(self.columns))
                    assert len(np.where(np.isnan(certain[i]) == False)[0]) + distr[i].d == len(self.columns)
                elif type(distr[i]) == sklearn.neighbors._kde.KernelDensity:
                    print(len(list(np.where(np.isnan(certain[i]) == False)[0])))
                    print(distr[i].n_features_in_)
                    print(len(self.columns))
                    assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + distr[i].n_features_in_ == len(self.columns)
                elif type(distr[i]) == list:
                    distr_list = distr[i]
                    # check if every list element is rv continuous or rv continuous frozen
                    print([l for l in distr_list if issubclass(type(l), scipy.stats.rv_continuous)
                                or issubclass(type(l), scipy.stats._distn_infrastructure.rv_continuous_frozen)])
                    assert len([l for l in distr_list if issubclass(type(l), 
                                                                    scipy.stats.rv_continuous)
                                or issubclass(type(l), 
                                              scipy.stats._distn_infrastructure.rv_continuous_frozen)]) == len(distr_list)
                    #erlaube hier auch kernel, brauche dafür eine zweite Liste
                    #ist das mit Christian abgeklärt?
                    ####################################################################################
                    # check if length of list is correct
                #FEHLT: LISTE VON KERNELS ERLAUBEN
                #bislang nur Liste von rv Distributions erlaubt 
                    assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + len(distr[i]) == len(self.columns)
                elif issubclass(type(distr[i]),
                                scipy.stats._multivariate.multi_rv_generic) or issubclass(type(distr[i]), 
                                                    scipy.stats._multivariate.multi_rv_frozen):
                   
                   
                   
                    assert len(list(np.where(np.isnan(certain[i]) == False)[0]))+distr[i].mean.shape[0] ==len(self.columns)
                 
                #check if it is 1D distr (one missing attribute in this row)
                elif issubclass(type(distr[i]),
                                scipy.stats.rv_continuous) or issubclass(type(distr[i]), 
                                scipy.stats._distn_infrastructure.rv_continuous_frozen):
                   
                    assert len(list(np.where(np.isnan(certain[i]) == False)[0])) == len(self.columns)-1
                else:
                    raise ValueError("Distribution type not supported")
         
            else:
               
                assert len(list(np.where(np.isnan(certain[i]) == False)[0])) == len(self.columns)

            #print("Certain data", certain[i][np.isnan(certain[i]) == False])
            if i in distr.keys():
                self.data.append(uframe_instance(uncertain_obj=distr[i],
                                             certain_data=certain[i][np.isnan(
                                                 certain[i]) == False],
                                             indices=[list(np.where(np.isnan(certain[i]) == True)[0]),
                                                      list(np.where(np.isnan(certain[i]) == False)[0])]))
            else:
                self.data.append(uframe_instance(uncertain_obj=None,
                                             certain_data=certain[i][np.isnan(
                                                 certain[i]) == False],
                                             indices=[list(np.where(np.isnan(certain[i]) == True)[0]),
                                                      list(np.where(np.isnan(certain[i]) == False)[0])]))
        return

 
    # append from a list of uframe_instance objects
    def append_from_uframe_instance(self, instances, colnames=None):

        if len(self.columns) > 0:

            dimensions = len(
                [instance.n_vars for instance in instances if instance.n_vars == len(self.columns)])
            if dimensions != len(instances):
                raise ValueError(
                    "Dimensions of new instances do not match dimension of uframe")

        # treat colnames parameter here in else case
        else:
            if colnames is None:
                self._columns = [*list(range(instances[0].n_vars))]
                self._colnames = {i: i for i in range(instances[0].n_vars)}
            else:
                if len(colnames) != instances[0].n_vars:
                    raise ValueError(
                        "Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name: i for i,
                                      name in enumerate(colnames)}

        self.data = self.data + instances

        return

    # samples: list of length (n_instances) of np.ndarray of shape (dim_samples, n_samples) oder listen von np.arrays (dim_samples,)
    # erstellt kernels und greift auf bereits existierende append-Funktion zurück
    # für scipy Gaussian und alle sklearn Kernels
    def append_from_samples(self, samples_list, kernel='stats.gaussian_kde', colnames=None, rownames=None):

        if len(samples_list) < 1:
            raise ValueError("No samples given")

        kernel_list = []

        for i, samples in enumerate(samples_list):

            if type(samples) == list:
                samples = np.array(samples).T

            if kernel == 'stats.gaussian_kde':
                kde = stats.gaussian_kde(values)
            elif kernel in ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']:
                samples = samples.T
                kde = KernelDensity(kernel=kernel, bandwidth=1.0).fit(samples)
            else:
                raise NotImplementedError("Given kernel does not exist")
            kernel_list.append(kde)

        if kernel == 'stats.gaussian_kde':
            self.append_from_scipy_kde(kernel_list, colnames=colnames)
        else:
            self.append_from_sklearn_kde(kernel_list, colnames=colnames)

        self.addRownames(kernel_list, rownames)
        return

    def addRownames(self, new, rownames):

        if rownames is not None:

            # check rownames for duplicates (also with previously existing rownames)
            if len(set(self._rows + rownames)) != len(self.rows) + len(rownames):
                raise ValueError("Duplicates among rownames")
            if len(rownames) != len(new):
                raise ValueError(
                    "Number of rownames given does not match number of rows given")

            self._rows = self._rows + rownames
            self._rownames.update(
                {rownames[i]: len(self.data)-len(new)+i for i in range(len(rownames))})
        else:

            self._rows = self.rows + [*list(range(len(self.data)-len(new), len(self.data)))]
            self._rownames.update({i: i for i in range(
                len(self.data)-len(new), len(self.data))})

        return

    def __repr__(self):

        return "Object of class uframe"

    def __str__(self):

        print("Object of class uframe with certain values:")
        print(self.array_rep())
        return ""


    def array_rep(self):

        x = np.zeros((len(self.data), len(self.columns)), dtype=np.float64)
        for i, instance in enumerate(self.data):
            x[i, instance.indices[1]] = instance.certain_data
            x[i, instance.indices[0]] = np.nan

        return x


    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, new_columns):

        if len(self.columns) > 0 and len(new_columns) != len(self._columns):
            raise ValueError(
                "Length of new column list does not match column number of uframe")

        if len(set(new_columns)) != len(new_columns):
            raise ValueError("New column list contains duplicates!")

        # update _colnames dictionary if it is not empty

        new_d = {}
        for i, new_name in enumerate(new_columns):

            new_d[new_name] = self._colnames.pop(self.columns[i])

        self._colnames = new_d
        self._columns = new_columns
        return

    @property
    def rows(self):
        return self._rows

    # rows always have to be integers so far, probably keep it that way like pandas
    @rows.setter
    def rows(self, new_rows):

        if len(self.rows) > 0 and len(new_rows) != len(self._rows):
            raise ValueError(
                "Length of new row list does not match row number of uframe")

        if len(set(new_rows)) != len(new_rows):
            raise ValueError("New row list contains duplicates")

        # check that all row indices are integers
        if len([index for index in new_rows if type(index) == int]) != len(new_rows):
            raise ValueError("Indices of rows have to be integers")

        new_d = {}
        for i, new_index in enumerate(new_rows):

            new_d[new_index] = self._rownames.pop(self.rows[i])

        self._rownames = new_d
        self._rows = new_rows

        return

    # function for reordering columns and rows
    # Parameter: Liste mit Teilmenge der Spalten in einer bestimmten Reihenfolge, interpretiere das als
    # neue Reihenfolge dieser Spalten, i.e., wenn bei Spalten [1,2,4] der Parameter [4,2,1] übergeben wird,
    # werden diese 3 Spalten so neu angeordnet, der Rest bleibt unverändert
    # dies wird nur für die Ausgabe in columns gespeichert (passe dann array_rep Funktion entsprechend an)
    # Frage: wird dann auf Input für append Funktionen entsprechend eingegangen und gemäß
    # _colnames vertauscht vor dem Speichern? Müsste aus Gründen der Benutzerfreundlichkeit eigentlich so passieren
    # würde in den append-Funktionen einem geeigneten Umsortieren gemäß self._colnames entsprechen
    # noch zu besprechen
    def reorder_columns(self):

        raise NotImplementedError()

    def mode(self):

        return np.concatenate([inst.mode() for inst in self.data], axis=0)

 
    def sample(self, n=1, seed=None):

        return np.concatenate([inst.sample(n, seed) for inst in self.data], axis=0)

    #does not work on uframe instance level yet
    def ev(self):
        
        return np.concatenate([inst.ev() for inst in self.data], axis=0)

    def get_dummies(self):
        print('pending')
        return

    def __getitem__(self, index):
        print('pending')
        return self.data[index]


#takes np array, randomly picks percentage of values p and introduces uncertainty there
#allow different kernels for the values given by mice, then use the mixed distr append function 
#allow scipy or sklearn kernels
#one multidimensional kernel is fitted for each row with missing values  
def uframe_from_array_mice(a: np.ndarray, p=0.1, mice_iterations=5, kernel="stats.gaussian_kde"):
    
    x, missing = generate_missing_values(a,p)
    
    distr={}
    
    #train mice imputation correctly 
    kds = mf.ImputationKernel(
        x,
        save_all_iterations=True,
        random_state=100)
    
    kds.mice(mice_iterations)
    for i in range(x.shape[0]):
        #print("Line", i)
        imp_distr= None 
        imp_arrays = []
        for j in range(x.shape[1]): 
            if np.isnan(x[i,j]):
              
                imp_values=[]
               
                for k in range(mice_iterations):
                    imput_x = kds.complete_data(iteration=k)
                    imp_values.append(imput_x[i,j])
                
                imp_value_arr = np.array(imp_values).reshape((1,mice_iterations))
                imp_arrays.append(imp_value_arr)
        
        if len(imp_arrays)==0:
            continue
        imp_array = np.concatenate(imp_arrays, axis=0)
        #print("Shape of imp_array", imp_array.shape)
        if kernel=="stats.gaussian_kde":
            kde = stats.gaussian_kde(imp_array)
            # print("Dimension of kde", kde.d)       
        else:
            imp_array = imp_array.T
            kde = KernelDensity(kernel=kernel, bandwidth=1.0).fit(imp_array)
            print("Dimension of kde", kde.n_features_in_)      
            
        imp_distr = kde 
                
        distr[i]= imp_distr
    
    u = uframe()
    u.append(new=[x, distr])
    
    return u

# add Gaussian noise of given std to chosen entries
# relative=True: multiply std with standard deviation of the column to get the std for a column
def uframe_noisy_array(a: np.ndarray, std=0.1, relative=False, unc_percentage=0.1):
    
    if relative==True:
        print("Get standard deviations of each column")
        stds = std*np.std(a, axis=1)
    else:
        stds = np.repeat(std, a.shape[1])
     
    print(stds)
    
    #delete unc_percentage of all values
    x, missing = generate_missing_values(a, unc_percentage)
    
    print(x, missing)
    #create suitable dictionary of distributions
    distr={}
    for i in range(x.shape[0]):
        if np.any(np.isnan(x[i,:]))==False:
            continue
        mean = a[i,:][np.where(np.isnan(x[i,:]) == True)[0]]
        cov = np.diag([stds[np.where(np.isnan(x[i,:]) == True)[0]]])
        
        print(mean, cov)
        
        distr[i]= scipy.stats.multivariate_normal(mean=mean, cov=cov)
        
    #generate uframe und use append function and return it 
    u = uframe()
    u.append(new=[x, distr])
    
    return u 

def generate_missing_values(complete_data, p):
    shape = complete_data.shape
    y = complete_data.copy()
    missing = np.random.binomial(1, p, shape)
    print(missing)
    y[missing.astype('bool')] = np.nan
    return y, missing

if __name__ == "__main__":
    a = uframe()
    # a.append_from_numpy(new=np.identity(3))

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

    c = uframe()
    c_array = np.array([[1, np.nan, 3], [np.nan, np.nan, 2]])

    kernel2 = stats.gaussian_kde(m1)

    kernel3 = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(values.T)

    distr1 = scipy.stats.norm()

    distr2 = scipy.stats.multivariate_normal()
    #c.append_from_mix_with_distr(certain=c_array, distr={0: kernel2, 1:kernel3})

    #have to check if that is supposed to work and fix code accordingly, if so 
    #c.append(new=[c_array, {0: distr1, 1:[kernel2,kernel2]}])

    d = uframe()
    d.append_from_rv_cont([distr1])
    d.append([distr1])
    d.append(distr2)
    d.append_from_rv_cont([[distr1], distr1])

    e = uframe()
    
    mv = scipy.stats.multivariate_normal(mean=np.array([1,1]))
    mv2 = scipy.stats.multivariate_normal(mean=np.array([1,1,1]))
    
    #e.append_from_rv_cont([mv2])
    #e.append_from_rv_cont([[distr1,distr1, distr1]])
    #e.append_from_rv_cont([[distr1, mv]])
    
    e.append(mv2)
    e.append([mv2])
    e.append([[distr1,distr1,distr1]])
    e.append([[distr1,distr1,distr1], mv2])

    #mixed mit RV Distributionen 
    #brauche geeignetes 3x3-Array für Test 
    
    #muss das später mit einer kompletten nan-Zeile testen 
    e = uframe()
    e_array = np.array([[1,np.nan, 3], [np.nan, 2, np.nan], [0, np.nan, np.nan]])
    
    e.append(new=[e_array, {0:distr1, 1: [distr1,distr1], 2: mv}])
    
    f = uframe()
    f.append([np.array([[1, np.nan],[1,1]]), {0:[distr1]}])
    
    g= uframe()
    g.append_from_rv_cont([distr1])
    
    #noch fehlend: Umstellung zu kategoriellen Variablen und entsprechende Anpassungen 
    #append Funktionen für entsprechende Mischungen (mix-Fall muss dann auch 1D kategorielle gemischt mit 1D RV erlauben)
    #fehlend: columns tauschen - da gibt es noch Dinge zu klären
    #fehlend:ev auf instance Ebene & Anpassungen, dass certain_data=[] erlaubt ist
    #fehlend: OHE, col_dtype Handling 
    #uframe aus np.array mit mice 
    
    #sampling aus u funktioniert aktuell nicht, mit Christian abklären, 
    #die Sample-Funktion benötigt eh ein Update nach categoricals
    #könnte Problem der eindimensionalen MV sein, MV über manche Zeilen und sonst feste Werte
    #können wir noch besprechen und klären 
    #auch mode geht hier nicht, Problem besprechen 
    a = np.array([[1,3.6,4.2], [2.2, 1, 3], [7,6,5], [8,8,2]])
    u = uframe_noisy_array(a, 0.1, True, 0.3)
