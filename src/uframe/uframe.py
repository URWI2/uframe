import numpy as np
#from src.uframe import uframe_instance
import scipy
from .uframe_instance import uframe_instance
from scipy import stats
from sklearn.neighbors import KernelDensity
import sklearn.neighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import miceforest as mf
import math
import pickle 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages

#SOLANGE DAS NICHT FUNKTIONIERT; MÜSSEN WERTE IN KATEGORIELLEN SPALTEN GANZZAHLIG SEIN
#nur Integer als Keys für kategorielle Verteilung zugelassen, muss append Funktion entsprechend anpassen 
#checke jeweils, ob schon cat value dict vorhanden, falls ja, ob es ergänzt werden muss
#falls kein cat value dict vorhanden, suche eine cat value Liste, die geeignet ist 

#expected value im kategoriellen Fall und geeignete Anpassung der get dummies Funktionen
#nach ev darf mit diesme Array einfach kein get dummies with array gecallt werden 

#self cat value dicts muss automatisch erstellt und geupdatet werden, damit ev so funktioniert 
#kann dann auch die OHE Funktionen damit anpassen 


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
        

    update():
        Update specific instance, not immediately relevant

    append(new = None):
        Append new rows to the data which can be either certain or uncertain.
        In place. New rows have to be numpy arrays at first, later add
        different sources of uncertain new rows.


    """

    def __init__(self,  new = None, colnames = None, rownames = None):

        self.data = []
        self._columns = []
        self._colnames = {}
        self._rows = []
        self._rownames = {}
        self._col_dtype = []

        if new is not None: 
            self.append(new = new, colnames = colnames, rownames = rownames)

            
            
            
   
    def analysis(self, true_data, save = False, **kwargs): 
        
        if save is not False:
              if not type (save) == str:
                  save  = 'analysis_uframe'
              
              pdf = PdfPages(str(save)+'.pdf') 
              _save = True 
          
              
          
          #Mode analysis
        mode = self.mode()
        for i in range(len(self._columns)):
            if self._col_dtype[i]== 'continuous':
             
                hist_fig, (hist_ax,hist_true) = plt.subplots(1, 2, figsize=(9, 3))
                hist_ax.set_title('Mode, var: '+ str(self._colnames[i]))
                    
                hist_ax.hist(mode[:,i],**kwargs)
                
                hist_true.set_title('True values, var:'+str(self._colnames[i]))
                hist_true.hist(true_data[:,i],**kwargs)
                
                
                if _save == True: 
                    hist_fig.savefig(pdf, format = 'pdf')        
                  
          
          #EV analysis
        ev = self.ev()
        for i in range(len(self._columns)):
            if self._col_dtype[i]== 'continuous':
              
                hist_fig, (hist_ax, hist_true) = plt.subplots(1, 2, figsize=(9, 3))
                hist_ax.set_title('EV, var:'+ str(self._colnames[i]))
                
                hist_true.set_title('True values, var:'+str(self._colnames[i]))
                hist_true.hist(true_data[:,i],**kwargs)
                
                
                hist_ax.hist(ev[:,i],**kwargs)
                if _save == True: 
                    hist_fig.savefig(pdf, format = 'pdf')        
                      
          

        for i in range(len(self._columns)):
            if self._col_dtype[i]== 'continuous':
                #Residues Mode - True 
                hist_fig, (hist_ax,table_ax) = plt.subplots(1, 2, figsize=(9, 3))
                hist_ax.set_title('Residue of Mode, var:'+ str(self._colnames[i]))
                    
                hist_ax.hist(mode[:,i]-true_data[:,i],**kwargs)
                table_ax.axis('tight')
                table_ax.axis('off')
                table_ax.table(analysis_table(true_data[:,i], ev[:,i]), loc = "center")
                
                
                if _save == True: 
                    hist_fig.savefig(pdf, format = 'pdf')        
                  
                #Residues EV-True
                hist_fig, (hist_ax,table_ax) = plt.subplots(1, 2, figsize=(9, 3))
                hist_ax.set_title('Residue of EV, var:'+ str(self._colnames[i]))
                residues = ev[:,i]-true_data[:,i]    
                hist_ax.hist(residues,**kwargs)
                table_ax.axis('tight')
                table_ax.axis('off')
                table_ax.table(analysis_table(true_data[:,i], ev[:,i]), loc = "center")
                                
                
                if _save == True: 
                    hist_fig.savefig(pdf, format = 'pdf')        
                                  
            
            #Accuracy in case of categorical
            """
            if self._col_dtype[i]== 'categorical':
                hist_fig, hist_ax = plt.subplots(1, 1, figsize=(9, 3))
                hist_ax.set_title('Accuracy of Mode, var:'+ str(self._colnames[i]))
                    
                hist_ax.hist(mode[:,i] == true_data[:,i],**kwargs)
                  
                if _save == True: 
                    hist_fig.savefig(pdf, format = 'pdf')        
            """
        
        pdf.close()
        plt.close()

 
    def append(self, new=None, colnames=None, rownames=None):

        if new is None:
            return

        if isinstance(new, np.ndarray):
            self.append_from_numpy(new, colnames, rownames)

        elif isinstance(new, scipy.stats._kde.gaussian_kde):
            self.append_from_scipy_kde([new], colnames)

        elif isinstance(new, sklearn.neighbors._kde.KernelDensity):
            self.append_from_sklearn_kde([new], colnames)

        elif isinstance(new, uframe_instance):
            self.append_from_uframe_instance([new], colnames)

        # one categorical distribution
        elif isinstance(new, dict):
            self.append_from_categorical([new], colnames)

        elif issubclass(type(new), scipy.stats.rv_continuous) or issubclass(type(new),
                                                                            scipy.stats._distn_infrastructure.rv_continuous_frozen):
            self.append_from_rv_cont([new], colnames)

        elif issubclass(type(new), scipy.stats._multivariate.multi_rv_generic) or issubclass(type(new),
                                                                                             scipy.stats._multivariate.multi_rv_frozen):
            self.append_from_rv_cont([new], colnames)

        elif isinstance(new, list):

            if len(new) == 4 and isinstance(new[0], np.ndarray) and isinstance(new[1], dict):
                #try:
                self.append_with_mixed_distributions(certain=new[0], continuous=new[1],
                                                         categorical=new[2], indices=new[3])
                #except BaseException:
                    #raise ValueError("Unsuitable parameter new given")

            if len(new) == 2 and isinstance(new[0], np.ndarray) and isinstance(new[1], dict):

                # check if dict contains continuous or categorical distributions
                if isinstance(new[1][list(new[1].keys())[0]], list) and isinstance(new[1][list(new[1].keys())[0]][0], dict):
                    self.append_with_mixed_categorical(new[0], new[1], colnames)
                else:
                    self.append_from_mix_with_distr(new[0], new[1], colnames)

            # one with two lists (or a list of lists of length 2) with only continuous und categorical variables

            if isinstance(new[0], scipy.stats._kde.gaussian_kde):
                self.append_from_scipy_kde(new, colnames)
            elif isinstance(new[0], sklearn.neighbors._kde.KernelDensity):
                self.append_from_sklearn_kde(new, colnames)
            elif isinstance(new[0], uframe_instance):
                #print("Entered uframe instance list append loop")
                self.append_from_uframe_instance(new, colnames)
            elif issubclass(type(new[0]), scipy.stats.rv_continuous) or issubclass(type(new[0]), scipy.stats._distn_infrastructure.rv_continuous_frozen):
                self.append_from_rv_cont(new, colnames)
            elif issubclass(type(new[0]), scipy.stats._multivariate.multi_rv_generic) or issubclass(type(new[0]),
                                                                                                    scipy.stats._multivariate.multi_rv_frozen):
                self.append_from_rv_cont(new, colnames)
            # Fall, wo Liste von Listen von 1D Distribs übergeben wird
            # checke, dass hier keine kategorischen Verteilungen übergeben werden
            elif isinstance(new[0], list):
                if len(new[0]) > 0 and not isinstance(new[0][0], dict):
                    self.append_from_rv_cont(new, colnames)
                elif len(new[0]) > 0 and isinstance(new[0][0], dict):
                    self.append_from_categorical(new, colnames)
        
        else: 
            raise ValueError("Unknown type of new")
            
        # sollte evtl. lieber die rownames in jeder append Funktion gesondert adressieren
        if isinstance(new, np.ndarray):
            return
        elif not isinstance(new, list):
            self.addRownames([new], rownames)
        elif isinstance(new[0], np.ndarray):
            self.addRownames(new[0], rownames)
        else:
            self.addRownames(new, rownames)

        return

    # append a numpy array with certain data (2D-array)
    def append_from_numpy(self, new=None, colnames=None, rownames=None):
        
        self.append_with_mixed_distributions(certain=new, continuous={}, categorical={},
                                             indices={}, colnames=colnames)
        
        if rownames is not None:

            # check rownames for duplicates (also with previously existing rownames)
            if len(set(self._rows + rownames)) != len(self.rows) + len(rownames):
                raise ValueError("Duplicates among rownames")
            if len(rownames) != len(new):
                raise ValueError(
                    "Number of rownames given does not match number of rows given")

            self._rows = self._rows + rownames
            self._rownames.update(
                {rownames[i]: len(self.data) - len(new) + i for i in range(len(rownames))})
        else:

            self._rows = self.rows + [*list(range(len(self.data) - len(new), len(self.data)))]
            # print("Rows", self.rows, "updated with", [*list(range(len(self.data) - len(new), len(self.data)))])
            self._rownames.update({i: i for i in range(
                len(self.data) - len(new), len(self.data))})
       
        return

    # distr list: list of list of dicts (1 list of dicts per instance)
    def append_from_categorical(self, distr_list, colnames=None):
        
        #need correct dimension from first element of distr_list
        dimension = len(distr_list[0])
        a = np.empty((len(distr_list),dimension))
        a[:]= np.nan
        self.append_with_mixed_distributions(certain=a, continuous={}, 
                                             categorical={i:distr_list[i] for i in range(len(distr_list))},
                                             indices={i:[[], [*list(range(dimension))]] for i in range(len(distr_list))})
        '''
        if len(self.columns) > 0:
            dimensions = len([len(l) for l in distr_list if len(l) == len(self.columns)])
            if dimensions != len(distr_list):
                raise ValueError("Dimension of list element does not match dimension of uframe")
            conts = [i for i in range(len(self.columns)) if self._col_dtype[i] == 'continuous']
            if len(conts) > 0:
                raise ValueError("Categorical distribution in continuous column")
        else:
            dimensions = len([len(l) for l in distr_list if len(l) == len(distr_list[0])])
            if dimensions != len(distr_list):
                raise ValueError("Distributions in list must have same dimension")

            if colnames is None:
                self._columns = [*list(range(len(distr_list[0])))]
                self._colnames = {i: i for i in range(len(distr_list[0]))}
            else:
                if len(colnames) != len(distr_list[0]):
                    raise ValueError("Length of column list does not match dim. of distributions")
                else:
                    self._columns = colnames
                    self._colnames = {name: i for i, name in enumerate(colnames)}
            self._col_dtype = len(self.columns) * ['categorical']

        for i, distr in enumerate(distr_list):
            self.data.append(uframe_instance(continuous=None, categorical=distr,
                                             certain_data=None,
                                             indices=[[], [], [*list(range(len(distr)))]]))

        return
    
    '''
    def append_from_scipy_kde(self, kernel_list, colnames=None):
        
        dimension = kernel_list[0].d
        a = np.empty((len(kernel_list), dimension))
        a[:]= np.nan
        self.append_with_mixed_distributions(certain=a, continuous={i:kernel_list[i] for i in range(len(kernel_list))}, 
                                             categorical={}, 
                                             indices={i:[[*list(range(dimension))],[]] for i in range(len(kernel_list))})

        '''
        if len(self.columns) > 0:

            dimensions = len(
                [kernel.d for kernel in kernel_list if kernel.d == len(self.columns)])
            if dimensions != len(kernel_list):
                raise ValueError(
                    "Dimension of list element does not match dimension of uframe")
            categoricals = [i for i in range(len(self.columns)) if self._col_dtype[i] == 'categorical']
            if len(categoricals) > 0:
                raise ValueError("Continuous kernel in categorical column")

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

            self._col_dtype = len(self.columns) * ['continuous']

        for i, kernel in enumerate(kernel_list):
            self.data.append(uframe_instance(continuous=kernel, certain_data=None,
                                             indices=[[], [*list(range(kernel.d))], []]))
         '''   
        return

    def append_from_sklearn_kde(self, kernel_list, colnames=None):
        
        dimension = kernel_list[0].n_features_in_
        a = np.empty((len(kernel_list), dimension))
        a[:]= np.nan
        self.append_with_mixed_distributions(certain=a, continuous={i:kernel_list[i] for i in range(len(kernel_list))}, 
                                             categorical={}, 
                                             indices={i:[[*list(range(dimension))],[]] for i in range(len(kernel_list))})
        
        '''
        if len(self.columns) > 0:

            dimensions = len(
                [kernel.n_features_in_ for kernel in kernel_list if kernel.n_features_in_ == len(self.columns)])
            if dimensions != len(kernel_list):
                raise ValueError(
                    "Dimension of list element does not match dimension of uframe")
            categoricals = [i for i in range(len(self.columns)) if self._col_dtype[i] == 'categorical']
            if len(categoricals) > 0:
                raise ValueError("Continuous kernel in categorical column")

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
            self._col_dtype = len(self.columns) * ['continuous']

        for i, kernel in enumerate(kernel_list):
            self.data.append(uframe_instance(continuous=kernel, certain_data=None,
                                             indices=[[], [*list(range(kernel.n_features_in_))], []]))
        '''

    # jeder Eintrag von distr_list muss entweder eine multivariate Verteilung über alle Variablen
    # oder eine Liste unabhängiger 1D-Verteilungen
    def append_from_rv_cont(self, distr_list, colnames=None):
        
        distr= distr_list[0]
        if issubclass(type(distr), scipy.stats._multivariate.multi_rv_generic) or issubclass(type(distr), scipy.stats._multivariate.multi_rv_frozen):
            dimension = distr.mean.shape[0]
        elif issubclass(type(distr), scipy.stats._distn_infrastructure.rv_continuous_frozen) or issubclass(type(distr), scipy.stats.rv_continuous):
            dimension = distr.rvs(size=1).shape[0]
        elif isinstance(distr, list):
            dimension = sum([d.rvs(size=1).shape[0] for d in distr])
        else:
            raise ValueError("Invalid argument in distribution list")
        
        a = np.empty((len(distr_list),dimension))
        a[:] = np.nan
        self.append_with_mixed_distributions(certain=a, 
                                             continuous = {i:distr_list[i] for i in range(len(distr_list))}, 
                                             categorical={}, 
                                             indices={i:[[*list(range(dimension))],[]] for i in range(len(distr_list))})
            
        
        return

    # assume dictionary with indices of incomplete lines as keys and scipy kdes as values
    # nan values for uncertain values in array certain
    def append_from_mix_with_distr(self, certain, distr, colnames=None):
        
        self.append_with_mixed_distributions(certain=certain, 
                                             continuous=distr, 
                                             categorical={}, 
                                             indices={i:[list(np.where(np.isnan(certain[i]))[0]),[]] for i in distr.keys()})
        
        return

    def append_with_mixed_categorical(self, certain, distr, colnames=None):
        
        self.append_with_mixed_distributions(certain=certain, continuous={}, 
                                             categorical=distr, 
                                             indices= {i:[[],list(np.where(np.isnan(certain[i]))[0])] for i in distr.keys()})
        
        return

    def append_with_mixed_distributions(self, certain, continuous, categorical, indices, colnames=None):

        # certain is an array with missing values indicated by nan values
        # continuous: dict, keys are indices of lines with missing values, values are MV distr., kernels or lists of 1D distr.
        # categorical: dict, keys are indices of lines with missing values, values are lists of dicts with categorical distributions for them
        # indices: dict, keys are indices of lines with missing values, values are lists of 2 lists, first list are indices of missing values with continuous distr, second with categorical
        # indices have to match the missing values in certain and the dimensions of the continuous and categorical distr.

        #print("Indices", indices)
        
        if len(self.columns) > 0:

            if len(self.columns) != certain.shape[1]:
                raise ValueError("Dimensions of new data do not match uframe")
            conts = [i for i in range(len(self.columns)) if self._col_dtype[i] == 'conts']
            for index in conts:
                if np.any(np.isnan(certain[:, index])):
                    raise ValueError("Categorical distribution in continuous column")
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

        self._col_dtype = []
        for col_index in range(len(self.columns)):
            if np.any(np.isnan(certain[:, col_index])):
                categoricals = [i for i in range(len(certain)) if i in indices.keys() and col_index in indices[i][1]]
                if len(categoricals) > 0:
                    conts = [i for i in range(len(certain)) if i in indices.keys() and col_index in indices[i][0]]
                    if len(conts) > 0:
                        raise ValueError("Continuous and categorical distr in same column")
                        certain_values = certain[:, col_index][np.isnan(certain[:, col_index] == False)]
                        if np.any(certain_values - np.floor(certain_values)):
                            raise ValueError("Float values and categorical distr in same column")
                    self._col_dtype.append('categorical')
                else:
                    self._col_dtype.append('continuous')
            else:
                self._col_dtype.append('continuous')

        # only for continuous case
        for i in range(len(certain)):
            if i not in continuous.keys():
                if i not in categorical.keys():
                    assert len(list(np.where(np.isnan(certain[i]) == False)[0])) == len(self.columns)
                    self.data.append(uframe_instance(continuous=None, categorical=None,
                                                     certain_data=certain[i, :],
                                                     indices=[list(np.where(np.isnan(certain[i]) == False)[0]),
                                                              [], []]))
                else:
                    assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + len(categorical[i]) == len(self.columns)
                    self.data.append(uframe_instance(continuous=None,
                                                     categorical=categorical[i],
                                                     certain_data=certain[i][np.isnan(
                                                         certain[i]) == False],
                                                     indices=[list(np.where(np.isnan(certain[i]) == False)[0]), [],
                                                              list(np.where(np.isnan(certain[i]))[0])]))

            else:
                if i not in categorical.keys():
                    if isinstance(continuous[i], scipy.stats._kde.gaussian_kde):

                        assert len(np.where(np.isnan(certain[i]) == False)[0]) + continuous[i].d == len(self.columns)
                    elif isinstance(continuous[i], sklearn.neighbors._kde.KernelDensity):

                        assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + continuous[i].n_features_in_ == len(self.columns)
                    elif isinstance(continuous[i], list):
                        distr_list = continuous[i]
                        # check if every list element is rv continuous or rv continuous frozen

                        assert len([l for l in distr_list if issubclass(type(l),
                                                                        scipy.stats.rv_continuous)
                                    or issubclass(type(l),
                                                  scipy.stats._distn_infrastructure.rv_continuous_frozen)]) == len(distr_list)

                        # check if length of list is correct
                        assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + len(continuous[i]) == len(self.columns)
                    elif issubclass(type(continuous[i]),
                                    scipy.stats._multivariate.multi_rv_generic) or issubclass(type(continuous[i]),
                                                                                              scipy.stats._multivariate.multi_rv_frozen):

                        assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + continuous[i].mean.shape[0] == len(self.columns)

                    # check if it is 1D distr (one missing attribute in this row)
                    elif issubclass(type(continuous[i]),
                                    scipy.stats.rv_continuous) or issubclass(type(continuous[i]),
                                                                             scipy.stats._distn_infrastructure.rv_continuous_frozen):

                        assert len(list(np.where(np.isnan(certain[i]) == False)[0])) == len(self.columns) - 1
                    else:
                        raise ValueError("Distribution type not supported")

                    self.data.append(uframe_instance(continuous=continuous[i],
                                                     certain_data=certain[i][np.isnan(
                                                         certain[i]) == False],
                                                     indices=[list(np.where(np.isnan(certain[i]) == False)[0]),
                                                              list(np.where(np.isnan(certain[i]))[0]), []]))
                if i in categorical.keys():

                    # check that the missing value indices and the distributions indices match
                    if set(np.where(np.isnan(certain[i, :]))[0]) != set(indices[i][0] + indices[i][1]):
                        raise ValueError("Indices of distributions do not match indices of missing values")
                    if isinstance(continuous[i], scipy.stats._kde.gaussian_kde):

                        assert len(np.where(np.isnan(certain[i]) == False)[0])
                        + continuous[i].d + len(categorical[i]) == len(self.columns)

                    elif isinstance(continuous[i], sklearn.neighbors._kde.KernelDensity):

                        left = len(list(np.where(np.isnan(certain[i]) == False)[0]))+ continuous[i].n_features_in_ + len(categorical[i])
                        right = len(self.columns)
                        assert left==right
                        
                    elif isinstance(continuous[i], list):
                        distr_list = continuous[i]
                       
                        assert len([l for l in distr_list if issubclass(type(l),
                                                                        scipy.stats.rv_continuous)
                                    or issubclass(type(l),
                                                  scipy.stats._distn_infrastructure.rv_continuous_frozen)]) == len(distr_list)

                        # check if length of list is correct
                        assert len(list(np.where(np.isnan(certain[i]) == False)[0]))
                        + len(continuous[i]) + len(categorical[i]) == len(self.columns)

                    elif issubclass(type(continuous[i]),
                                    scipy.stats._multivariate.multi_rv_generic) or issubclass(type(continuous[i]),
                                                                                              scipy.stats._multivariate.multi_rv_frozen):

                        assert len(list(np.where(np.isnan(certain[i]) == False)[0]))
                        + continuous[i].mean.shape[0] + len(categorical[i]) == len(self.columns)

                    elif issubclass(type(continuous[i]),
                                    scipy.stats.rv_continuous) or issubclass(type(continuous[i]),
                                                                             scipy.stats._distn_infrastructure.rv_continuous_frozen):

                        assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + len(categorical[i]) == len(self.columns)

                    else:
                        raise ValueError("Distribution type not supported")

                    self.data.append(uframe_instance(continuous=continuous[i],
                                                     categorical=categorical[i],
                                                     certain_data=certain[i][np.isnan(
                                                         certain[i]) == False],
                                                     indices=[list(np.where(np.isnan(certain[i]) == False)[0]),
                                                              indices[i][0], indices[i][1]]))

        return

    # append from a list of uframe_instance objects
    def append_from_uframe_instance(self, instances, colnames=None):

        if len(self.columns) > 0:

            dimensions = len(
                [len(instance) for instance in instances if len(instance) == len(self.columns)])
            if dimensions != len(instances):
                raise ValueError("Dimensions of new instances do not match dimension of uframe")
            categoricals = [i for i in range(len(self.columns)) if self._col_dtype[i] == 'categorical']
            for index in categoricals:
                # check if that index is continuous or a certain float value in any of the new uframe_instances
                for instance in instances:
                    if index in instance.indices[1]:
                        raise ValueError("Continuous distribution in categorical column")
                    if index in instance.indices[0]:
                        certain_value = instance.certain_data[instance.indices[0].index(index)]
                        if certain_value - math.floor(certain_value) > 0:
                            raise ValueError("Float value in categorical column")

        # treat colnames parameter here in else case
        else:
            if colnames is None:
                self._columns = [*list(range(len(instances[0])))]
                self._colnames = {i: i for i in range(len(instances[0]))}
            else:
                if len(colnames) != len(instances[0]):
                    raise ValueError("Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name: i for i,
                                      name in enumerate(colnames)}
            self._col_dtype = len(self.columns) * ['continuous']

        # print("Data", self.data)
        # print("Instances", instances)
        self.data = self.data + instances
        # print("After appending", self.data)

        return

    # samples: list of length (n_instances) of np.ndarray of shape (dim_samples, n_samples) oder listen von np.arrays (dim_samples,)
    # erstellt kernels und greift auf bereits existierende append-Funktion zurück
    # für scipy Gaussian und alle sklearn Kernels
    def append_from_samples(self, samples_list, kernel='stats.gaussian_kde', colnames=None, rownames=None):

        if len(samples_list) < 1:
            raise ValueError("No samples given")

        kernel_list = []

        for i, samples in enumerate(samples_list):

            if isinstance(samples, list):
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
                {rownames[i]: len(self.data) - len(new) + i for i in range(len(rownames))})
        else:

            self._rows = self.rows + [*list(range(len(self.data) - len(new), len(self.data)))]
            # print("Rows", self.rows, "updated with", [*list(range(len(self.data) - len(new), len(self.data)))])
            self._rownames.update({i: i for i in range(
                len(self.data) - len(new), len(self.data))})

        return

    def __len__(self):
        return len(self.data)

    def __repr__(self):

        return "Object of class uframe"

    def __str__(self):

        print("Object of class uframe with certain values:")
        print(self.array_rep())
        return ""

    def array_rep(self):

        x = np.zeros((len(self.data), len(self.columns)), dtype=np.float64)
        for i, instance in enumerate(self.data):
            x[i, instance.indices[0]] = instance.certain_data
            x[i, instance.indices[1]] = np.nan
            x[i, instance.indices[2]] = np.nan

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
        if len([index for index in new_rows if isinstance(index, int)]) != len(new_rows):
            raise ValueError("Indices of rows have to be integers")

        new_d = {}
        for i, new_index in enumerate(new_rows):

            new_d[new_index] = self._rownames.pop(self.rows[i])

        self._rownames = new_d
        self._rows = new_rows

        return

    def get_row(self, key):
        return self._rownames[key]

    @property
    def col_dtype(self):

        # könnte das hier ändern, sodass es die Reihenfolge gemäß self.columns zurückgibt
        # noch nicht relevant, aber falls Ändern der Spaltenreihenfolge später möglich ist
        # nutze dafür dann self._colnames für Zuordnungen
        return self._col_dtype

    # set type of a column, if new type matches the values and distributions in that column
    # not tested yet
    def set_col_dtype(self, column, col_dtype):

        if col_dtype not in ['continuous', 'categorical']:
            raise ValueError("Invalid col_dtype")

        if column not in self._colnames.keys():
            raise ValueError("Column does not exist")

        col_index = self._colnames[column]
        if self._col_dtype[col_index] == col_dtype:
            #print("Column already has desired col_dtype")
            return
        if col_dtype == 'continuous':
            for instance in self.data:
                if col_index in instance.indices[2]:
                    raise ValueError("Categorical distributions in this column, cannot make it continuous")
        elif col_dtype == 'categorical':
            for instance in self.data:
                if col_index in instance.indices[1]:
                    raise ValueError("Continuous distributions in this column, cannot make it categorical")
                elif col_index in instance.indices[0]:
                    certain_value = instance.certain_data[instance.indices[0].index(col_index)]
                    if certain_value - math.floor(certain_value) > 0:
                        raise ValueError("Float value in column, cannot make it categorical")
        self._col_dtype[col_index] = col_dtype

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

    def sample(self, n=1, seed=None, threshold = 1):

        return np.concatenate([inst.sample(n = n, seed = seed, threshold = threshold) for inst in self.data], axis=0)

    def ev(self):
        
        evs = [inst.ev() for inst in self.data]
        if 'categorical' not in self._col_dtype:
            
            #print([(ev, type(ev)) for ev in evs])
            return np.concatenate([ev.reshape((1, len(ev))) for ev in evs], axis=0)
        else:
            #print("Evs", evs)
            
            #conts = [ev[0].shape for ev in evs]
            cats = []
            conts = []
            for i, ev in enumerate(evs):
                if type(ev) is list:
                    cats.append(ev[1])
                    conts.append(ev[0])
                else:
                    cats.append([])
                    conts.append(ev)
            
           # print("Length of cats", len(cats), cats)
            
            #print("Ev 0 list", [ev[0] for ev in evs])
            #print([ev[0].shape for ev in evs])
            #print("Conts", conts)
            #print([cont.shape for cont in conts])
            conts = np.concatenate([cont.reshape((1, len(cont))) for cont in conts], axis=0)
            #print("Conts", conts, conts.shape)
            ohes = {}
            for i in range(len(self.columns)):
                if self._col_dtype[i]=='categorical':
                
                    #have some known values in conts as well as nan values
                    
                    #have to iterate over lines
                    #know all levels of this variable from cat value dict
                    #need to make sure it exists 
                    levels = list(self.cat_value_dicts[i].keys())
                    m = np.zeros((conts.shape[0], len(levels)))
                    for j in range(conts.shape[0]):
                        #print("i", i, "j", j, "Indices", self.data[j].indices)
                        if np.isnan(conts[j,i]):
                            index = self.data[j].indices[2].index(i)
                            #print("I", i, "j", j, "Index", index)
                            for key in cats[j][index].keys():
                                #print("Cat value dict for column i", self.cat_value_dicts[i])
                                #print("Key", key)
                                m[j,int(self.cat_value_dicts[i][key])]= cats[j][index][key]
                        else:
                            m[j, int(conts[j,i])]=1
                    
                    ohes[i]= m
                    #print("Shape of ohes i", ohes[i].shape)
                
                #get new array from conts and the m arrays in ohes 
            ev_array = None
            for i in range(len(self.columns)):
                #print("i", i)
                if ev_array is None:
                    if self._col_dtype[i]=='categorical':
                        ev_array = ohes[i]
                    else:
                        ev_array = conts[:,i].reshape((conts.shape[0],1))
                else:
                    if self._col_dtype[i]=='categorical':
                        ev_array = np.concatenate((ev_array, ohes[i]), axis=1)
                    else:
                        ev_array = np.concatenate((ev_array, conts[:,i].reshape((conts.shape[0],1))), axis=1)
            
            return ev_array 
                
                    
    # performs One Hot Encoding of categorical columns after filling uncertain values by sampling/ mode/ ev
    # return columns in some way?
    def get_dummies(self, method='sample', samples_per_distrib=1):

        # do sampling, ev, modal
        # then iterate over the columns and for categorical
        if method == 'sample':
            x = self.sample(samples_per_distrib, None)
        elif method == 'mode':
            x = self.mode()
        elif method == 'ev':
            #expected value already incorporates ohe in categorical case 
            x = self.ev()
            return x 
        else:
            raise ValueError("Need method for treating uncertain values")
        
        
        return self.get_dummies_for_array(x)
        
    #gets dummies for given array x, which has to match the column number, no extra method there
    def get_dummies_for_array(self, x):
        
        x_ohe = None
        for i in range(x.shape[1]):
            if self._col_dtype[i] == 'categorical':

                integer_encoded = x[:,i]
                
                if hasattr(self, 'cat_values') and i in self.cat_values.keys():
                    categories = self.cat_values[i]
                else:
                    categories = []
                    
                for j in range(len(self.data)):
                    if i in self.data[j].indices[0]:
                        categories.append(self.data[j].certain_data[self.data[j].indices[0].index(i)])
                    elif i in self.data[j].indices[2]:

                        categories = categories + [float(key) for key in 
                                                   self.data[j].categorical[self.data[j].indices[2].index(i)].keys()]
                 
                categories = [np.array(list(set(categories)))]

                #print("Categories", categories)
                onehot_encoder = OneHotEncoder(sparse_output=False, categories= categories)
                integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
                onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
                
                # print("One hot encoded shape", onehot_encoded.shape)
                if x_ohe is None:
                    x_ohe = onehot_encoded
                else:
                    x_ohe = np.concatenate((x_ohe, onehot_encoded), axis=1)

            else:
                if x_ohe is None:
                    x_ohe = x[:, i].reshape((len(x), 1))
                else:
                    x_ohe = np.concatenate((x_ohe, x[:, i].reshape((len(x), 1))), axis=1)

        return x_ohe
           
    def make_cat_value_dicts(self):
        
        self.cat_value_dicts = {}
        
        for i in range(len(self.columns)):
            
            #print("I", i)
            
            if self._col_dtype[i]=='categorical':
            
            #check if self has a cat values attribute
                if hasattr(self, 'cat_values') and i in self.cat_values.keys():
                    categories = self.cat_values[i]
                else:
                    categories = []
                
                for j in range(len(self.data)):
                    if i in self.data[j].indices[0]:
                        categories.append(self.data[j].certain_data[self.data[j].indices[0].index(i)])
                    elif i in self.data[j].indices[2]:
    
                        categories = categories + [float(key) for key in 
                                               self.data[j].categorical[self.data[j].indices[2].index(i)].keys()]
             
                categories = list(set(categories))
                #print("Categories", categories)
                self.cat_value_dicts[i]= {categories[i]:i for i in range(len(categories))}
            
            else:
                self.cat_value_dicts[i]= {}
        
        return 
    
    #new stuff as parameter: do not think so, would have to find out what new uframe instances are there
    def update_cat_value_dicts(self):
        
        #check for each categorical column if all its values are already in the cat value dict 
        #if not, append new keys with them and give them the correct index 
        
        
        return 
            
    
    def __getitem__(self, index):
        return uframe(new = self.data[index], colnames = self._columns, rownames = self._rows[index])

    # TO DO: function which takes i,j and returns element of uframe (need marginal distributions for that)

    def add_cat_values(self, cat_value_dict):
        
        if hasattr(self, 'cat_values'):
            self.cat_values.update(cat_value_dict)
        else:
            self.cat_values = cat_value_dict
        
        self.update_cat_value_dicts()
        
        return 
    
     
    def save(self, name="saved_uframe.pkl"):
            
        with open(name, 'wb') as f:
            pickle.dump(self,f)
        return 
     

def analysis_table(true, preds): 
    residues = true - preds
    true_q= np.array([sum(true[k]<true)/len(true) for k in range(len(true))])
    new_q = np.array([sum(preds[k]<true)/len(true) for k in range(len(true))])

    return [["Var",str(round(np.var(residues),2))],
            ["MAE", str(round(np.mean(np.abs(residues)),2))],
            ["RMSE", str(round(np.sqrt(np.mean(residues**2)),2))],
            ["Diff Quantile", str(round(np.mean(np.abs(true_q - new_q)),2))]]
            

if __name__ == "__main__":
    a = uframe()

    def measure(n):
        m1 = np.random.normal(size=n)
        m2 = np.random.normal(scale=0.5, size=n)
        return m1 + m2, m1 - m2

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
    # c.append_from_mix_with_distr(certain=c_array, distr={0: kernel2, 1:kernel3})

    d = uframe()
    d.append_from_rv_cont([distr1])
    d.append([distr1])
    d.append(distr2)
    d.append_from_rv_cont([[distr1], distr1])

    e = uframe()

    mv = scipy.stats.multivariate_normal(mean=np.array([1, 1]))
    mv2 = scipy.stats.multivariate_normal(mean=np.array([1, 1, 1]))

    # e.append_from_rv_cont([mv2])
    # e.append_from_rv_cont([[distr1,distr1, distr1]])
    # e.append_from_rv_cont([[distr1, mv]])

    e.append(mv2)
    e.append([mv2])
    e.append([[distr1, distr1, distr1]])
    e.append([[distr1, distr1, distr1], mv2])

    # mixed mit RV Distributionen

    e = uframe()
    e_array = np.array([[1, np.nan, 3], [np.nan, 2, np.nan], [0, np.nan, np.nan]])

    e.append(new=[e_array, {0: distr1, 1: [distr1, distr1], 2: mv}])

    f = uframe()
    f.append([np.array([[1, np.nan], [1, 1]]), {0: [distr1]}])

    g = uframe()
    g.append_from_rv_cont([distr1])

    # fehlend: columns tauschen - da gibt es noch Dinge zu klären

    a = np.array([[1, 3.6, 4.2], [2.2, 1, 3], [7, 6, 5], [8, 8, 2]])
    u = uframe_noisy_array(a, 0.1, True, 0.3)

    h = uframe()
    h.append([[{0: 0.3, 1: 0.4, 2: 0.3}], [{0: 0.8, 1: 0.2}]])

    h = uframe()
    a = np.array([[1, 2, np.nan], [1, 3, np.nan]])
    h.append([a, {0: [{0: 0.7, 1: 0.3}], 1: [{0: 0.1, 3: 0.9}]}])
    x_ohe = h.get_dummies()
    
    from scipy.stats import gamma, norm
    uncertain = [uframe_instance(certain_data = np.array([2.1]), continuous = [norm(0.2,1), gamma(0.3)], indices = [[1],[0,2],[]]),
                 uframe_instance(continuous = [norm(0.1,1), norm(0.3,0.7), gamma(1)], indices = [[],[1,0,2],[]])]
    data = uframe()
    data.append(uncertain)
    print(data.sample())
