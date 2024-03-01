"""
uframe_instance class used in uframe to depict a single uncertain data instance.

"""
from scipy.integrate import nquad
from scipy.integrate import cumulative_trapezoid
import scipy
import numpy as np
import sklearn
import warnings
import numpy.typing as npt
from typing import Optional, List, Dict


class uframe_instance():
    """
    uframe_instance
    ===============

    The `uframe_instance` class is designed to represent a single uncertain data instance within the `uframe` package.
    It encapsulates continuous, categorical, and certain data along with their associations, providing a structured way
    to handle uncertain data. This class is a fundamental part of the `uframe` package, offering a structured approach to handling and analyzing
    a single uncertain data instance. It abstracts the complexity of managing uncertainty
    and its computation, providing a unified interface for sampling and optimization tasks.


    Parameters
    ----------------------
    certain_data : Optional[npt.ArrayLike]
        Certain data instances represented as a 1D numpy array.
    continuous : scipy.stats._kde.gaussian_kde | scipy.stats and other compatible types
        Describes the uncertain variables of the instance using Scipy Kernel Density Estimation or similar models.
    categorical : Optional[List[Dict[str, float]]]
        A list of dictionaries, each representing the probability distribution of a categorical variable.
    indices : Optional[List[List[int]]]
        Specifies the order in which samples and mode values should be returned. It is shaped as
        [[indices certain], [indices continuous], [indices categorical]].


    Returns
    -------
    out: uframe_instance
        An uframe instance object satisfying the specified requirements.


    """

    def __init__(self, certain_data: Optional[npt.ArrayLike] = None, continuous=None,
                 categorical: Optional[List[Dict[str, float]]] = None, indices: Optional[List[List[int]]] = None):
        """ Constructor Method
        """

        certain_data = np.array([]) if certain_data is None else certain_data
        certain_data = np.array([certain_data]) if type(
            certain_data) == list else certain_data
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
            if np.any(certain_data.imag != 0):
                raise TypeError("Certain Data must be real valued")
            self.n_certain = len(certain_data)

        if categorical is not None:
            self.__init_categorical(categorical)

        self.indices = indices
        if indices is None:
            self.indices = [[*range(self.n_certain)],
                            [*range(self.n_certain, self.n_certain +
                                    self.n_continuous)],
                            [*range(self.n_certain + self.n_continuous, self.n_continuous + self.n_certain + self.n_categorical)]]

        assert self.__check_indices(
            continuous, self.certain_data, self.indices)

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
            warnings.warn(
                "The provided KDE does not has an gaussian or tophat kernel, this might result in Errors")
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

    def var(self, n: int = 50, seed: Optional[int] = None):
        """
        Calculates the variance of the uncertain data instance by sampling.


        Parameters
        ----------
        n : int, optional
            The number of samples to generate from the uncertain data instance. The default value is 50.


        Returns
        -------
        float
            The variance of the uncertain data instance. For multi-dimensional data, this will be a vector of variances for each dimension, depending on the implementation specifics.

        Raises
        ------
        ValueError
            If the instance's data does not allow for variance computation due to missing or invalid data types or distributions.

        Notes
        -----
        - This method assumes that all components of the uncertain data instance are appropriately initialized and that their variances can be meaningfully computed and aggregated.

        Example
        -------
        Assuming `uframe_inst` is an instance of `uframe_instance`:

        >>> variance = uframe_inst.var()
        >>> print(variance)
        # Output: The variance of the uncertain data instance.

        """
        return np.var(self.sample(n=n, seed=seed), axis=0)

    # mode functions
    def mode(self, **kwargs):
        """
        Calculates and returns the mode value of the uncertain instance.

        For non-parametric distributions, this method utilizes an optimizer to find the mode of the instance.
        The mode represents the value with the highest probability density.

        Returns
        -------
        np.array
            The mode value(s) of the instance. The shape and size of the returned array depend on the underlying
            uncertainty distribution and the dimensions of the uncertain data. For continuous data, this typically
            means returning the point(s) in the input space that correspond to the peak(s) of the probability density
            function.

        Raises
        ------
        NotImplementedError
            If the method for finding the mode is not implemented for the type(s) of uncertainty encapsulated by
            the instance. This can occur if the `continuous` attribute does not support an efficient way to calculate
            or estimate the mode.

        Notes
        -----
        - The mode calculation is particularly relevant for continuous uncertain data, where it signifies the
          most likely value within the distribution. For categorical data, the mode would correspond to the
          category with the highest probability.
        - The implementation of this method may vary depending on the specific types of uncertainty (continuous,
          categorical) involved and the models used to represent them. It may involve numerical optimization
          techniques, analytical solutions, or approximations.
        - The efficiency and accuracy of the mode calculation can be influenced by the complexity of the uncertainty
          model and the dimensionality of the data.
        """
        if not self._mode_calculated():
            self.__mode = self.__align(self.__mode_continuous(
                **kwargs), self.__mode_categorical())
        return self.__mode

    def _mode_calculated(self):
        return hasattr(self, f'_{self.__class__.__name__}__mode')

    def _set_mode(self, mode):
        self.__mode = mode

    def __mode_continuous(self, **kwargs):
        return np.array([])

    def __mode_scipy_kde(self, **kwargs):
        opt = scipy.optimize.basinhopping(
            lambda x: -self.continuous.pdf(x), np.zeros(self.n_continuous), **kwargs)
        return opt.x.reshape(1, -1)

    def __mode_sklearn_kde(self, **kwargs):
        opt = scipy.optimize.basinhopping(lambda x: -self.continuous.score_samples(
            x.reshape(1, -1)), np.zeros(self.n_continuous), **kwargs)
        return opt.x.reshape(1, -1)

    def __mode_scipy_rv_c(self, **kwargs):
        if (issubclass(type(self.continuous), scipy.stats._multivariate.multi_rv_generic) or
                issubclass(type(self.continuous), scipy.stats._multivariate.multi_rv_frozen)):

            opt = scipy.optimize.basinhopping(
                lambda x: -self.continuous.pdf(x), self.continuous.mean, **kwargs)
            return opt.x.reshape(1, -1)

        opt = scipy.optimize.basinhopping(
            lambda x: -self.continuous.pdf(x), self.continuous.mean(), **kwargs)
        return opt.x.reshape(1, -1)

    def __mode_dist_list(self):

        opt = [scipy.optimize.basinhopping(
            lambda x: -dist.pdf(x), dist.mean()).x for dist in self.continuous]
        opt = np.array(opt)

        return opt.reshape(1, -1)

    def __mode_categorical(self):
        if self.n_categorical == 0:
            return np.array([])
        return np.array([list(dist.keys())[np.argmax([*dist.values()])] for dist in self.categorical]).reshape(1, -1)

    # sampling functions
    def sample(self, n: int = 1, seed: Optional[int] = None, threshold: Optional[float] = 1):
        """
        Samples `n` instances from the uncertain data instance.

        This method generates samples from the uncertain data instance represented by the `uframe_instance` object. It utilizes the underlying uncertainty distributions to produce samples that are consistent with the uncertainty model. The method can handle continuous, categorical, and certain data, generating a comprehensive sample that reflects all aspects of the data instance's uncertainty.

        Parameters
        ----------
        n : int, optional
            The number of samples to generate from the uncertain data instance. The default value is 1.
        seed : int, optional
            The seed for the random number generator to ensure reproducibility of the samples. If not provided, the sampling process will be stochastic, potentially leading to different results on each invocation.
        threshold: float, optional
            If below 1, only the samples with the highest probability density are selected. Parameter defines percentage of samples chosen. Can be interpreted as cut-off of an pdf.
        Returns
        -------
        np.ndarray
            An array of sampled values. The shape of the array depends on the number of samples (`n`) requested. Each row in the output array corresponds to a single sample drawn from the uncertain data instance.

        Raises
        ------
        ValueError
            If the continuous uncertainty object is not recognized or supported by the method.

        Notes
        -----
        - The method integrates over the continuous and categorical uncertainties, as well as the certain data, to generate a composite sample that mirrors the structure of the `uframe_instance`.
        - The sampling process considers the indices and associations specified in the `uframe_instance` object to ensure that the sampled values are aligned with the respective continuous, categorical, and certain data components.

        Example
        -------
        Assuming an instance `uframe_inst` of `uframe_instance` has been properly initialized:

        >>> samples = uframe_inst.sample(n=10, seed=42)
        >>> print(samples)
        # This will print an array of 10 samples drawn from the `uframe_inst` according to its uncertainty model.

        """

        if threshold == 1:
            return self.__align(self.sample_continuous(n, seed), self.sample_categorical(n, seed), n)
        else:

            samples = self.__align(self.sample_continuous(
                n, seed), self.sample_categorical(n, seed), n)

            pdfs = self.pdf(samples)

            sort_ind = np.argsort(pdfs, axis=0)
            sort_ind = np.squeeze(sort_ind[:])
            sort_ind = sort_ind[sort_ind < round(n*threshold)]

            return samples[sort_ind, :]

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
        sampels = [dist.rvs(size=n, random_state=seed)
                   for dist in self.continuous]
        return np.column_stack(sampels)

    def ev(self, n: Optional[int] = 50, seed: Optional[int] = None):
        """
        Calculate the expected value of the uncertain instance.

        This method generates samples from the uncertain instance and computes their mean. If the instance has categorical data, the expected value for categorical variables is set to None, and categorical distributions are appended separately.

        Parameters
        ----------
        n : Optional[int], default=50
            The number of samples to generate for calculating the expected value. A higher number of samples may lead to a more accurate estimation but will require more computation.
        seed : Optional[int], default=None
            Seed for the random number generator to ensure reproducibility. If None, the randomness is unpredictable.

        Returns
        -------
        np.array | List[Union[np.array, List[Dict[str, float]]]]
            The expected value of the uncertain instance. For purely numeric data, this is a numpy array of the mean values. For instances with categorical data, the return value is a list containing the numpy array of mean values for continuous and certain data (with categorical means set to None) and the list of categorical distributions.

        Notes
        -----
        - The expected value for categorical data is not directly computed since categorical data represents discrete categories without a direct notion of 'average'. The method returns the categorical distributions themselves for further analysis.
        - This method caches its result. Subsequent calls with the same parameters will return the cached value without recomputation.

        Examples
        --------
        >>> u_instance = uframe_instance(certain_data=np.array([1, 2, 3]),
                                         continuous=scipy.stats.norm(
                                             loc=0, scale=1),
                                         categorical=[{"cat1": 0.5, "cat2": 0.5}])
        >>> u_instance.ev()
        [array([1., 2., 3., mean of samples]), [{'cat1': 0.5, 'cat2': 0.5}]]
        """
        if hasattr(self, "_uframe_instance__ev"):
            return self.__ev

        self.__ev = (self.sample(n, seed)).mean(axis=0)
        if self.n_categorical > 0:
            self.__ev[self.indices[2]] = None
            self.__ev = [self.__ev, self.categorical]

        return self.__ev

    def __check_categorical(self):

        for d in self.categorical:
            assert isinstance(d, dict)
            assert all([type(key) == int for key in list(d.keys())]
                       ), "Keys of categorical uncertain object have to be integer"

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

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) for the given input(x).

        This method calculates the PDF values for each input in 'x' considering only continuous uncertainties within the instance. 

        Parameters
        ----------
        x : np.array | List
            The input value(s) for which the CDF values should be computed. Can be a list or a numpy array. The input should match the structure and dimensions of the uncertain data instance, i.e., it should have the same number of elements as there are uncertain and certain variables in the instance.

        Returns
        -------
        np.array
            An array of CDF values corresponding to each input in 'x'.

        Notes
        -----
        - The method automatically handles inputs in list format by converting them to numpy arrays.

        Examples
        --------
        >>> u_instance = uframe_instance(certain_data=np.array([1, 2, 3]),
                                         continuous=scipy.stats.norm(loc=0, scale=1))
        >>> u_instance.cdf([0, 1, 2, 1])
        array([[PDF value]])

        >>> u_instance.pdf(np.array([[0, 1, 2], [3, 4, 5]]))
        array([[PDF value for [0, 1, 2]],
               [PDF value for [3, 4, 5]]])
        """
        if type(x) == list:
            x = np.array(x)

        x = x.reshape([-1, self.__len__()])
        ret = []
        for elem in x:

            cont = self._cdf_continuous(elem)
            ret.append(cont)

        return np.array(ret)

    def _cdf_continuous(self, x):

        if self.n_continuous == 0:
            return [1]

        if (isinstance(self.continuous, scipy.stats._kde.gaussian_kde) or
                isinstance(self.continuous, sklearn.neighbors._kde.KernelDensity)):
            start = self.ev(100)[self.indices[1]] - 3 * \
                self.var(100)[self.indices[1]]
            end = x.reshape(-1)[self.indices[1]]

            def wrapper(*args):
                return self.pdf_continuous(np.array(args))

            return nquad(wrapper, [(start[i], end[i]) for i in range(len(start))])[0]

        if (issubclass(type(self.continuous), scipy.stats.rv_continuous) or
            issubclass(type(self.continuous), scipy.stats._distn_infrastructure.rv_continuous_frozen) or
            issubclass(type(self.continuous), scipy.stats._multivariate.multi_rv_generic) or
                issubclass(type(self.continuous), scipy.stats._multivariate.multi_rv_frozen)):
            return self.continuous.cdf(x)

        if isinstance(self.continuous, list):
            return [self.continuous[i].pdf(x.reshape(-1)[i]) for i in range(self.n_continuous)]

        raise ValueError("Unknown continuous uncertainty object")

    def pdf(self, k):
        """
        Compute the probability density function (PDF) for the given input(s).

        This method calculates the PDF values for each input in 'k' considering all uncertainties (continuous, categorical) within the instance. It leverages the 'pdf_elementwise' method to compute the PDF for each element.

        Parameters
        ----------
        k : np.array | List
            The input value(s) for which the PDF should be computed. Can be a list or a numpy array. The input should match the structure and dimensions of the uncertain data instance, i.e., it should have the same number of elements as there are uncertain and certain variables in the instance.

        Returns
        -------
        np.array
            An array of PDF values corresponding to each input in 'k'.

        Notes
        -----
        - The method automatically handles inputs in list format by converting them to numpy arrays.
        - It reshapes the input 'k' to ensure it matches the expected dimensionality of the uncertain instance, facilitating element-wise PDF computation.
        - The 'pdf_elementwise' method is called internally to compute the PDF for individual elements based on their uncertainty type (continuous or categorical).

        Examples
        --------
        >>> u_instance = uframe_instance(certain_data=np.array([1, 2, 3]),
                                         continuous=scipy.stats.norm(loc=0, scale=1))
        >>> u_instance.pdf([0, 1, 2, 1])
        array([[PDF value]])

        >>> u_instance.pdf(np.array([[0, 1, 2, 1], [3, 4, 5, 1]]))
        array([[PDF value for [0, 1, 2, 1]],
               [PDF value for [3, 4, 5, 1]]])
        """
        if type(k) == list:
            k = np.array(k)

        k = k.reshape([-1, self.__len__()])
        ret = np.array([self.pdf_elementwise(elem) for elem in k])

        return ret.reshape(-1, 1)

    def pdf_elementwise(self, k):
        cat = self.pdf_categorical(k[self.indices[2]])
        cont = self.pdf_continuous(k[self.indices[1]])

        if isinstance(cont, np.float64):
            cont = [cont]

        if isinstance(cat, np.float64):
            cat = [cat]

        return np.prod([*cat, *cont])

    def pdf_categorical(self, k):

        if self.n_categorical == 0:
            return np.array([1, 1])
        return np.prod([np.max([*dist.values()]) for dist in self.categorical]).reshape(1, -1)

    def pdf_continuous(self, k):
        if self.n_continuous == 0:
            return np.array([1, 1])

        if isinstance(self.continuous, scipy.stats._kde.gaussian_kde):
            return self.continuous.pdf(k)

        if isinstance(self.continuous, sklearn.neighbors._kde.KernelDensity):
            return np.exp(self.continuous.score_samples(k.reshape(1, -1)))

        if (issubclass(type(self.continuous), scipy.stats.rv_continuous) or
            issubclass(type(self.continuous), scipy.stats._distn_infrastructure.rv_continuous_frozen) or
            issubclass(type(self.continuous), scipy.stats._multivariate.multi_rv_generic) or
                issubclass(type(self.continuous), scipy.stats._multivariate.multi_rv_frozen)):
            return self.continuous.pdf(k)

        if isinstance(self.continuous, list):
            return [self.continuous[i].pdf(k[i]) for i in range(self.n_continuous)]

        raise ValueError("Unknown continuous uncertainty object")

    def __align(self, u_continuous, u_categorical, n=1):

        ret = np.zeros(
            (n, self.n_certain + self.n_categorical + self.n_continuous))
        ret[:, self.indices[0]] = self.certain_data
        ret[:, self.indices[1]] = np.array(u_continuous)
        ret[:, self.indices[2]] = u_categorical
        return ret
