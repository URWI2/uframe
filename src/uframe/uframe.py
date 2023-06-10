import numpy as np
from uframe_instance import uframe_instance
import scipy
from scipy import stats
from sklearn.neighbors import KernelDensity
import sklearn.neighbors
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import miceforest as mf
import math
from copy import deepcopy

# make only a private attribute _col_dtype, which is settled automatically
# check for contradictions with categorical variables
# each column has to be either categorical or continuous, including and checking that will be important
# in previous append functions without categorical types, a check will be needed for dtypes of columns as well
# but that can be done properly after everything works and a col_dtype attribute is introduced

# function for printing col_dtypes: for all columns, use _colnames to get right number and print out dtypes
# set coltype(columns, types): erlaube Listen oder Spaltennamen und Datentypen zum Setzen
# defaults: continuous für continuous und certain data, categorical für kategoriell
'''
neue append Funktionen:
- Liste von stetigen Verteilungen mit Liste von kat. Verteilungen und Indexinformationen (evtl. für alle Typen von RV & Kernel anpassen)
- 4er-Pack: Array, stetiges Dict., kat. Dict, Indexinformationen
-Liste gemischter stetiger Verteilungen/ Kerne (i.e., allgemeine Mischung aus append_scipy_kde, sklearn_kde, rv_cont): evtl.

'''

# append Funktion mit Liste von kat., Liste von kernel/ stetig (mehrere Typen) und Indexinformation
# append Funktion mit allen 3 Typen gemischt


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
        self._col_dtype = []

        return

    # checking for what new represents is not ideal as of now, probably need more checks for wrong values of new
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

            if len(new) == 4 and isinstance(new[0], np.array) and isinstance(new[1], dict):
                try:
                    self.append_with_mixed_distributions(certain=new[0], continuous=new[1],
                                                         categorical=new[2], indices=new[3])
                except BaseException:
                    raise ValueError("Unsuitable parameter new given")

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

        if len(self.columns) > 0:

            if new.shape[1] != len(self.columns):
                raise ValueError("New data does not match shape of uframe")

            else:
                categoricals = [i for i in range(len(self.columns)) if self._col_dtype[i] == 'categorical']
                for index in categoricals:
                    if np.any(new[:, index] - np.floor(new[:, index])):
                        raise ValueError("Float values in categorical column")
                for i in range(new.shape[0]):
                    self.data.append(uframe_instance(continuous=None, certain_data=new[i, :],
                                                     indices=[[*list(range(new.shape[1]))], [], []]))
        else:
            for i in range(new.shape[0]):
                #print("Append line", i)
                self.data.append(uframe_instance(continuous=None, certain_data=new[i, :], indices=[
                                 [*list(range(new.shape[1]))], [], []]))

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

            self._col_dtype = len(self.columns) * ['continuous']

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

    def append_from_scipy_kde(self, kernel_list, colnames=None):

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

        return

    def append_from_sklearn_kde(self, kernel_list, colnames=None):

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
                             for distr in distr_list if isinstance(distr, list)]

        d_list = dimensions_mv + dimensions_rv + dimensions_dlists
        if len(self.columns) > 0:
            matches = len([i for i in d_list if i == len(self.columns)])
            if matches != len(distr_list):
                raise ValueError(
                    "Dimension of distributions must match uframe dimension")
            categoricals = [i for i in range(len(self.columns)) if self._col_dtype[i] == 'categorical']
            if len(categoricals) > 0:
                raise ValueError("Continuous distribution in categorical column")

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
            self._col_dtype = len(self.columns) * ['continuous']

        for i, distr in enumerate(distr_list):

            if issubclass(type(distr), scipy.stats._distn_infrastructure.rv_continuous_frozen) or issubclass(type(distr),
                                                                                                             scipy.stats.rv_continuous):
                assert distr.rvs(size=1).shape[0] == len(self.columns)
            elif issubclass(type(distr), scipy.stats._multivariate.multi_rv_generic) or issubclass(type(distr),
                                                                                                   scipy.stats._multivariate.multi_rv_frozen):

                assert distr.mean.shape[0] == len(self.columns)
            # checke, dass die Liste nur 1D-Verteilungen enthält
            # erlaube Liste mit multivariater Verteilung zunächst nicht, mit Christian absprechen, ob seine Klasse das kann
            else:
                list_len = len([l for l in distr if issubclass(type(l), scipy.stats.rv_continuous)
                                or issubclass(type(l), scipy.stats._distn_infrastructure.rv_continuous_frozen)])
                assert list_len == len(self.columns)
                assert len(distr) == len(self.columns)
            self.data.append(uframe_instance(continuous=distr, certain_data=None,
                                             indices=[[], [*list(range(d_list[0]))], []]))

        return

    # assume dictionary with indices of incomplete lines as keys and scipy kdes as values
    # nan values for uncertain values in array certain
    def append_from_mix_with_distr(self, certain, distr, colnames=None):

        if len(self.columns) > 0:

            if len(self.columns) != certain.shape[1]:
                raise ValueError("Dimensions of new data do not match uframe")
            categoricals = [i for i in range(len(self.columns)) if self._col_dtype[i] == 'categorical']
            for index in categoricals:
                if np.any(np.isnan(certain[:, index])):
                    raise ValueError("Continuous distribution in categorical column")
                certain_values = certain[:, index][np.isnan(certain[:, index] == False)]
                if np.any(certain_values - np.floor(certain_values)):
                    raise ValueError("Float values in categorical column")

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
            self._col_dtype = len(self.columns) * ['continuous']

        for i in range(len(certain)):

            if i in distr.keys():

                if isinstance(distr[i], scipy.stats._kde.gaussian_kde):

                    assert len(np.where(np.isnan(certain[i]) == False)[0]) + distr[i].d == len(self.columns)
                elif isinstance(distr[i], sklearn.neighbors._kde.KernelDensity):

                    assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + distr[i].n_features_in_ == len(self.columns)
                elif isinstance(distr[i], list):
                    distr_list = distr[i]
                    # check if every list element is rv continuous or rv continuous frozen
                    print([l for l in distr_list if issubclass(type(l), scipy.stats.rv_continuous)
                           or issubclass(type(l), scipy.stats._distn_infrastructure.rv_continuous_frozen)])
                    assert len([l for l in distr_list if issubclass(type(l),
                                                                    scipy.stats.rv_continuous)
                                or issubclass(type(l),
                                              scipy.stats._distn_infrastructure.rv_continuous_frozen)]) == len(distr_list)

                    # check if length of list is correct

                    # bislang nur Liste von rv Distributions erlaubt
                    assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + len(distr[i]) == len(self.columns)
                elif issubclass(type(distr[i]),
                                scipy.stats._multivariate.multi_rv_generic) or issubclass(type(distr[i]),
                                                                                          scipy.stats._multivariate.multi_rv_frozen):

                    assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + distr[i].mean.shape[0] == len(self.columns)

                # check if it is 1D distr (one missing attribute in this row)
                elif issubclass(type(distr[i]),
                                scipy.stats.rv_continuous) or issubclass(type(distr[i]),
                                                                         scipy.stats._distn_infrastructure.rv_continuous_frozen):

                    assert len(list(np.where(np.isnan(certain[i]) == False)[0])) == len(self.columns) - 1
                else:
                    raise ValueError("Distribution type not supported")

            else:

                assert len(list(np.where(np.isnan(certain[i]) == False)[0])) == len(self.columns)

            if i in distr.keys():
                self.data.append(uframe_instance(continuous=distr[i],
                                                 certain_data=certain[i][np.isnan(
                                                     certain[i]) == False],
                                                 indices=[list(np.where(np.isnan(certain[i]) == False)[0]),
                                                          list(np.where(np.isnan(certain[i]))[0]), []]))
            else:
                self.data.append(uframe_instance(continuous=None,
                                                 certain_data=certain[i, :],
                                                 indices=[list(np.where(np.isnan(certain[i]) == False)[0]),
                                                          [], []]))
        return

    def append_with_mixed_categorical(self, certain, distr, colnames=None):
        # certain is an array with missing values indicated by nan values
        # distr: dict, keys are indices of lines with missing values, values are lists of dicts with categorical distributions for them

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

            # col_dtype erfodert zusätzlichen Check hier: sind nicht-ganzzahlige Werte und kat. Verteilungen in derselben Spalte
            self._col_dtype = []
            for col_index in range(len(self.columns)):
                if np.any(np.isnan(certain[:, col_index])):
                    print("Col index", col_index)
                    certain_values = certain[:, col_index][np.isnan(certain[:, col_index] == False)]
                    if np.any(certain_values - np.floor(certain_values)):
                        raise ValueError("Float values and categorical distr in same column")
                    else:
                        print("Appended categorical")
                        self._col_dtype.append('categorical')
                        print("Col dtype", self._col_dtype)
                else:
                    self._col_dtype.append('continuous')

        for i in range(len(certain)):
            if i not in distr.keys():
                assert len(list(np.where(np.isnan(certain[i]) == False)[0])) == len(self.columns)
                self.data.append(uframe_instance(continuous=None, categorical=None,
                                                 certain_data=certain[i, :],
                                                 indices=[list(np.where(np.isnan(certain[i]) == False)[0]),
                                                          [], []]))
            else:
                assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + len(distr[i]) == len(self.columns)
                self.data.append(uframe_instance(continuous=None,
                                                 categorical=distr[i],
                                                 certain_data=certain[i][np.isnan(
                                                     certain[i]) == False],
                                                 indices=[list(np.where(np.isnan(certain[i]) == False)[0]), [],
                                                          list(np.where(np.isnan(certain[i]))[0])]))
        return

    # not tested yet
    def append_with_mixed_distributions(self, certain, continuous, categorical, indices, colnames=None):

        # certain is an array with missing values indicated by nan values
        # continuous: dict, keys are indices of lines with missing values, values are MV distr., kernels or lists of 1D distr.
        # categorical: dict, keys are indices of lines with missing values, values are lists of dicts with categorical distributions for them
        # indices: dict, keys are indices of lines with missing values, values are lists of 2 lists, first list are indices of missing values with continuous distr, second with categorical
        # indices have to match the missing values in certain and the dimensions of the continuous and categorical distr.

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
                categoricals = [i for i in range(len(certain)) if col_index in indices[i][1]]
                if len(categoricals) > 0:
                    conts = [i for i in range(len(certain)) if col_index in indices[i][0]]
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

                        assert len(list(np.where(np.isnan(certain[i]) == False)[0]))
                        + continuous[i].n_features_in_ + len(categorical[i]) == len(self.columns)

                    elif isinstance(continuous[i], list):
                        distr_list = continuous[i]
                        # check if every list element is rv continuous or rv continuous frozen

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
            print("Column already has desired col_dtype")
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

    def sample(self, n=1, seed=None):

        return np.concatenate([inst.sample(n, seed) for inst in self.data], axis=0)

    # does not work on uframe instance level yet
    def ev(self):

        return np.concatenate([inst.ev() for inst in self.data], axis=0)

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
            x = self.ev()
        else:
            raise ValueError("Need method for treating uncertain values")

        x_ohe = None
        for i in range(x.shape[1]):
            if self._col_dtype[i] == 'categorical':

                # brauchen wir hier label encoding überhaupt? Nur ganzzahlige Werte
                label_encoder = LabelEncoder()
                integer_encoded = label_encoder.fit_transform(x[:, i])
                # print(integer_encoded)
                categories = []
                for j in range(len(self.data)):
                    if i in self.data[j].indices[0]:
                        categories.append(self.data[j].certain_data[self.data[j].indices[0].index(i)])
                    elif i in self.data[j].indices[2]:

                        categories = categories + [float(key) for key in 
                                                   self.data[j].categorical[self.data[j].indices[2].index(i)].keys()]
                 
                categories = [np.array(list(set(categories)))]

                # print("Categories", categories)
                onehot_encoder = OneHotEncoder(sparse=False, categories= categories)
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

    def __getitem__(self, index):
        return self.data[index]

    # TO DO: function which takes i,j and returns element of uframe (need marginal distributions for that)


# takes np array, randomly picks percentage of values p and introduces uncertainty there
# allow different kernels for the values given by mice, then use the mixed distr append function
# allow scipy or sklearn kernels
# one multidimensional kernel is fitted for each row with missing values
def uframe_from_array_mice(a: np.ndarray, p=0.1, mice_iterations=5, kernel="stats.gaussian_kde",
                           scaler= "min_max"):

    x, missing = generate_missing_values(a, p)

    distr = {}

    # train mice imputation correctly
    kds = mf.ImputationKernel(
        x,
        save_all_iterations=True,
        random_state=100)

    kds.mice(mice_iterations)
    for i in range(x.shape[0]):
        # print("Line", i)
        imp_distr = None
        imp_arrays = []
        for j in range(x.shape[1]):
            if np.isnan(x[i, j]):

                imp_values = []

                for k in range(mice_iterations):
                    imput_x = kds.complete_data(iteration=k)
                    imp_values.append(imput_x[i, j])

                imp_value_arr = np.array(imp_values).reshape((1, mice_iterations))
                imp_arrays.append(imp_value_arr)

        if len(imp_arrays) == 0:
            continue
        
        
        # for i, arr in enumerate(imp_arrays):
        #     scaler= MinMaxScaler()
        #     arr = scaler.fit_transform(arr)
        #     imp_arrays[i]= arr
        
        # print("Imp arrays after scaling")
        
        imp_array = np.concatenate(imp_arrays, axis=0)
        scaler = MinMaxScaler()
        imp_array = (scaler.fit_transform(imp_array.T)).T
        #print("Shape of imp array", imp_array.shape)

        if kernel == "stats.gaussian_kde":
            kde = stats.gaussian_kde(imp_array)

        else:
            imp_array = imp_array.T
            kde = KernelDensity(kernel=kernel, bandwidth=1.0).fit(imp_array)
            #print("Dimension of kde", kde.n_features_in_)

        imp_distr = kde

        distr[i] = imp_distr

    print(x)
    scaler= MinMaxScaler()
    x = scaler.fit_transform(x)
    #a[missing==1]=np.nan
    u = uframe()
    u.append(new=[x, distr])

    return u

#TO DO: version of uframe_from_array_mice for categorical imputation/ mix of categorical and continuous 

# add Gaussian noise of given std to chosen entries
# relative=True: multiply std with standard deviation of the column to get the std for a column
def uframe_noisy_array(a: np.ndarray, std=0.1, relative=False, unc_percentage=0.1):

    if relative:
        print("Get standard deviations of each column")
        stds = std * np.std(a, axis=1)
    else:
        stds = np.repeat(std, a.shape[1])

    print(stds)

    # delete unc_percentage of all values
    x, missing = generate_missing_values(a, unc_percentage)

    print(x, missing)
    # create suitable dictionary of distributions
    distr = {}
    for i in range(x.shape[0]):
        if np.any(np.isnan(x[i, :])) == False:
            continue
        mean = a[i, :][np.where(np.isnan(x[i, :]))[0]]
        cov = np.diag([stds[np.where(np.isnan(x[i, :]))[0]]])

        print(mean, cov)

        distr[i] = scipy.stats.multivariate_normal(mean=mean, cov=cov)

    # generate uframe und use append function and return it
    u = uframe()
    u.append(new=[x, distr])

    return u


def generate_missing_values(complete_data, p):
    shape = complete_data.shape
    y = complete_data.copy()
    missing = np.random.binomial(1, p, shape)
    
    y[missing.astype('bool')] = np.nan
    return y, missing


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
    a = np.array([[1, 2, np.nan], [np.nan, 3, np.nan]])
    h.append([a, {0: [{0: 0.7, 1: 0.3}], 1: [{0: 0.3, 1: 0.1, 2: 0.6}, {0: 0.1, 3: 0.9}]}])
    x_ohe = h.get_dummies()
