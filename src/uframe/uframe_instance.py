"""
uframe_instance class used in uframe to depict a single uncertain data instance.

"""
import scipy
import numpy as np
import sklearn
import warnings
import numpy.typing as npt
from typing import Optional, List, Dict


class uframe_instance():
    """
       A class used to represent an singel uncertain data instance.

       ...

       Attributes
       ----------
       continuous : class
           A class describing the underlying uncertainty.
       certain_data : np.array
           Numpy array of certain values
       indices : [list,list]
           associations of indices to continuous and certain data

       Methods
       -------
       sample(n = 1, seed = None)
           Samples n samples from the uncertain instance

       mode()
           Uses an optimizer to finde the mode value of this instance

    """

    def __init__(self, certain_data: Optional[npt.ArrayLike] = None, continuous=None,
                 categorical: Optional[List[Dict[str, float]]] = None, indices: Optional[List[List[int]]] = None):
        """
        Parameters
        ----------
        continuous : scipy.stats._kde.gaussian_kde | scipy.stats
            Scipy Kernel Density Estimation or other, that describes the ucnertain Variables of the instance.
        certain_data 1D np.array
            Certain data instances.
        categorical_uncertainty: list of dictionaries
            List of dictionaries with each dictionary representing the probability distribution of a categorical variable.

        indices : list
            list of indices which indicates the order in which samples and mode values should be returned.
        """

        certain_data = np.array([]) if certain_data is None else certain_data
        certain_data = np.array([certain_data]) if type(certain_data) == list else certain_data
        self.certain_data = certain_data

        self.indices = indices
        self.n_continuous = 0
        self.n_certain = 0
        self.n_categorical = 0

        if continuous is not None:
            if isinstance(continuous, scipy.stats._kde.gaussian_kde):
                self.__init_scipy_kde(continuous)
            elif isinstance(continuous, sklearn.neighbors._kde.KernelDensity):
                self.__init_sklearn_kde(continuous)
            elif (issubclass(type(continuous), scipy.stats.rv_continuous) or
                  issubclass(type(continuous), scipy.stats._distn_infrastructure.rv_continuous_frozen) or
                  issubclass(type(continuous), scipy.stats._multivariate.multi_rv_generic) or
                  issubclass(type(continuous), scipy.stats._multivariate.multi_rv_frozen)):
                self.__init_scipy_rv_c(continuous)

            elif isinstance(continuous, list):
                self.__init_list(continuous)

            else:
                raise ValueError("Unknown continuous uncertainty object")

        if certain_data is not None:
            assert type(certain_data) in [np.ndarray, np.array]
            self.n_certain = len(certain_data)

        if categorical is not None:
            self.__init_categorical(categorical)

        self.indices = indices
        if indices is None:
            self.indices = [[*range(self.n_certain)],
                            [*range(self.n_certain, self.n_certain + self.n_continuous)],
                            [*range(self.n_certain + self.n_continuous, self.n_continuous + self.n_certain + self.n_categorical)]]

        assert self.__check_indices(continuous, self.certain_data, self.indices)

    def __str__(self):
        return ("Uncertain data instance")

    def __repr__(self):
        return ("Uncertain data instance")

    def __len__(self):
        return self.n_categorical + self.n_certain + self.n_continuous

    # Initialization functions
    def __init_scipy_kde(self, kernel):
        self.continuous = kernel
        self.sample_continuous = self.__sample_scipy_kde
        self.__mode_continuous = self.__mode_scipy_kde
        self.n_continuous = self.continuous.d

    def __init_sklearn_kde(self, kernel):
        self.continuous = kernel
        self.sample_continuous = self.__sample_sklearn_kde
        self.__mode_continuous = self.__mode_sklearn_kde
        self.n_continuous = self.continuous.n_features_in_

        if self.continuous.get_params()["kernel"] not in ["gaussian", "tophat"]:
            warnings.warn("The provided KDE does not has an gaussian or tophat kernel, this might result in Errors")
            delattr(self, "sample")

    def __init_scipy_rv_c(self, rv):
        self.continuous = rv
        self.sample_continuous = self.__sample_scipy_rv_c
        self.__mode_continuous = self.__mode_scipy_rv_c
        self.n_continuous = rv.dim if hasattr(rv, "dim") else 1

    def __init_list(self, distributions):
        for i in distributions:
            assert (issubclass(type(i), scipy.stats.rv_continuous) or
                    issubclass(type(i), scipy.stats._distn_infrastructure.rv_continuous_frozen))

        self.continuous = distributions
        self.sample_continuous = self.__sample_dist_list
        self.__mode_continuous = self.__mode_dist_list
        self.n_continuous = len(distributions)

    def __init_categorical(self, categorical):
        self.categorical = categorical
        assert self.__check_categorical()
        self.n_categorical = len(self.categorical)

    # mode functions
    def mode(self):
        if not self._mode_calculated():
            self.__mode =  self.__align(self.__mode_continuous(), self.__mode_categorical())
        return self.__mode
    
    def _mode_calculated(self): 
        return hasattr(self, f'_{self.__class__.__name__}__mode')
    
    def _set_mode(self, mode): 
        self.__mode = mode
        
    def __mode_continuous(self):
        return np.array([])

    def __mode_scipy_kde(self):
        opt = scipy.optimize.basinhopping(lambda x: -self.continuous.pdf(x), np.zeros(self.n_continuous))
        return opt.x.reshape(1, -1)

    def __mode_sklearn_kde(self):
        opt = scipy.optimize.basinhopping(lambda x: -self.continuous.score_samples(x.reshape(1, -1)), np.zeros(self.n_continuous))
        return opt.x.reshape(1, -1)

    def __mode_scipy_rv_c(self):
        if (issubclass(type(self.continuous), scipy.stats._multivariate.multi_rv_generic) or
                issubclass(type(self.continuous), scipy.stats._multivariate.multi_rv_frozen)):

            opt = scipy.optimize.basinhopping(lambda x: -self.continuous.pdf(x), self.continuous.mean)
            return opt.x.reshape(1, -1)

        opt = scipy.optimize.basinhopping(lambda x: -self.continuous.pdf(x), self.continuous.mean())
        return opt.x.reshape(1, -1)

    def __mode_dist_list(self):

        opt = [scipy.optimize.basinhopping(lambda x: -dist.pdf(x), dist.mean()).x for dist in self.continuous]
        opt = np.array(opt)

        return opt.reshape(1, -1)

    def __mode_categorical(self):
        if self.n_categorical == 0:
            return np.array([])
        return np.array([list(dist.keys())[np.argmax([*dist.values()])] for dist in self.categorical]).reshape(1, -1)

    # sampling functions
    def sample(self, n: int = 1, seed: Optional[int] = None, threshold: Optional[float] = 1):
        if threshold == 1:
            return self.__align(self.sample_continuous(n, seed), self.sample_categorical(n, seed), n)
        else: 
            
            samples = self.__align(self.sample_continuous(n, seed), self.sample_categorical(n, seed), n)
            
            pdfs = self.pdf(samples)
            
            sort_ind = np.argsort(pdfs, axis = 0)
            sort_ind = np.squeeze(sort_ind[:])
            sort_ind = sort_ind[sort_ind < round(n*threshold)]
                
            return samples[sort_ind,:]             
                
                
                
    def sample_categorical(self, n: int = 1, seed: Optional[int] = None):
        if self.n_categorical == 0:
            return np.array([])
        return np.array([self.__sample_categorical_dist(dist, n, seed) for dist in self.categorical]).transpose()

    def __sample_categorical_dist(self, dist, n, seed: Optional[int] = None):
        return np.random.choice(list(dist.keys()), size=n, p=[*dist.values()])

    def sample_continuous(self, n: int = 1, seed: Optional[int] = None):
        return np.array([])

    def __sample_scipy_kde(self, n: int = 1, seed: Optional[int] = None):
        return self.continuous.resample(n, seed=seed).transpose()

    def __sample_sklearn_kde(self, n: int = 1, seed: Optional[int] = None):
        return self.continuous.sample(n_samples=n, random_state=seed)

    def __sample_scipy_rv_c(self, n: int = 1, seed: Optional[int] = None):
        if n > 1:
            return self.continuous.rvs(size=n, random_state=seed).reshape(n, self.n_continuous)

        return self.continuous.rvs(size=n, random_state=seed)

    def __sample_dist_list(self, n: int = 1, seed=None):
        sampels = [dist.rvs(size=n, random_state=seed) for dist in self.continuous]
        return np.column_stack(sampels)

    def ev(self, n: Optional[int] = 50, seed: Optional[int] = None):

        if hasattr(self,"_uframe_instance__ev"):
            return self.__ev

        self.__ev = (self.sample(n, seed)).mean(axis=0)
        if self.n_categorical >0: 
            self.__ev[self.indices[2]] = None
            self.__ev = [self.__ev, self.categorical]
        
        return self.__ev

    def __check_categorical(self):
        
        for d in self.categorical:
            assert isinstance(d, dict)
            assert all([type(key) == int for key in list(d.keys())]), "Keys of categorical uncertain object have to be integer"
            
        for d in self.categorical:
            assert sum(d.values()) == 1

        return True

    def __check_indices(self, continuous, certain_data, indices):
        assert isinstance(indices, list)
        assert len(indices) == 3

        comp_indices = [*indices[0], *indices[1], *indices[2]]
        for i in range(len(self)):
            assert i in comp_indices

        if certain_data is None:
            assert len(indices[0]) == 0

        if len(indices[1]) == 0 and continuous is None:
            return True

        if isinstance(continuous, scipy.stats._kde.gaussian_kde):
            return len(indices[1]) == continuous.dataset.shape[0]

        if isinstance(continuous, sklearn.neighbors._kde.KernelDensity):
            return len(indices[1]) == continuous.n_features_in_

        if (issubclass(type(continuous), scipy.stats.rv_continuous) or
                issubclass(type(continuous), scipy.stats._distn_infrastructure.rv_continuous_frozen)):
            return len(indices[1]) == 1

        return True


    def pdf(self, k): 
        if type(k) == list: 
            k = np.array(k)
        
        k = k.reshape([-1,self.__len__()])
        ret = np.array([self.pdf_elementwise(elem) for elem in k])
        
        return ret.reshape(-1, 1)
    
    
    def pdf_elementwise(self,k): 
        cat = self.pdf_categorical(k[self.indices[2]])
        cont = self.pdf_continuous(k[self.indices[1]])
        
        return np.prod([*cat,*cont])
        
    def pdf_categorical(self,k): 
        
        if self.n_categorical == 0:
            return np.array([1,1])
        return np.prod([np.max([*dist.values()]) for dist in self.categorical]).reshape(1, -1)

    def pdf_continuous(self,k):
        if self.n_continuous == 0: 
            return np.array([1,1])
        
        if isinstance(self.continuous, scipy.stats._kde.gaussian_kde):
            return self.continuous.pdf(k)
        
        if isinstance(self.continuous, sklearn.neighbors._kde.KernelDensity):
            return np.exp(self.continuous.score_samples(k.reshape(1,-1)))
            
        if (issubclass(type(self.continuous), scipy.stats.rv_continuous) or
              issubclass(type(self.continuous), scipy.stats._distn_infrastructure.rv_continuous_frozen) or
              issubclass(type(self.continuous), scipy.stats._multivariate.multi_rv_generic) or
              issubclass(type(self.continuous), scipy.stats._multivariate.multi_rv_frozen)):
            return self.continuous.pdf(k)
        
        if isinstance(self.continuous, list):
            return [self.continuous[i].pdf(k[i]) for i in range(self.n_continuous)]
        
        raise ValueError("Unknown continuous uncertainty object")
        
    def __align(self, u_continuous, u_categorical, n=1):

        ret = np.zeros((n, self.n_certain + self.n_categorical + self.n_continuous))
        ret[:, self.indices[0]] = self.certain_data
        ret[:, self.indices[1]] = np.array(u_continuous)
        ret[:, self.indices[2]] = u_categorical
        return ret
