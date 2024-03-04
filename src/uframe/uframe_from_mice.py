import numpy as np
from sklearn.preprocessing import MinMaxScaler
import miceforest as mf
from scipy import stats
from sklearn.neighbors import KernelDensity
from uframe import uframe
import random


def generate_missing_values(complete_data, p, seed, method='binomial'):
    shape = complete_data.shape
    data_out = complete_data.copy()
    np.random.seed(seed)
    random.seed(seed)
    if p == 0:
        missing = None

    elif method == 'binomial':
        missing = np.random.binomial(1, p, shape)

    elif method == 'attributes':

        n_choose = round(p*shape[0])
        missing = np.zeros(shape)
        for i in range(shape[1]):
            missing[random.sample(range(shape[0]), n_choose), i] = 1

    elif method == 'instances':

        n_choose = round(p*shape[1])
        missing = np.zeros(shape)
        for i in range(shape[0]):
            missing[i, random.sample(range(shape[1]), n_choose)] = 1

    else:
        raise ValueError("Unknown method")
    data_out = data_out.astype(np.float64)
    data_out[missing.astype('bool')] = np.nan

    return data_out, missing


def add_bias(old_values, bias, bias_method):

    new_values = old_values

    if bias_method == "fix":

        new_values = new_values + bias

    elif bias_method == "gaussian":

        for i in range(len(new_values)):
            new_values[i] = new_values[i] + random.normalvariate(bias, 1)

    elif bias_method == "even":

        for i in range(len(new_values)):
            new_values[i] = new_values[i] + \
                random.uniform(-abs(bias), abs(bias))

    return new_values


def uframe_from_array_sim(X: np.ndarray, p=0.5,
                          missing_method='binomial',
                          bias=0.5,
                          bias_method='even',
                          dist_method='gaussian',
                          std=0.1,
                          std_method='fix',
                          seed=None):
    """
    Create a uframe object from a numpy array by simulating uncertain data.

    This function simulates uncertainty within the given numpy array `X` by introducing variability based on a specified proportion `p`. The resulting data, characterized by simulated uncertainty, is then encapsulated into a `uframe` object for further analysis and processing.

    Parameters
    ----------
    X : np.ndarray
        The input data array to simulate uncertainty within. Should be a 2D numpy array where rows represent individual observations and columns represent features.
    p : float, default=0.5
        The proportion of uncertainty to simulate in the data. This could define the variance of the noise introduced or the proportion of values to replace with simulated uncertainty, depending on the implementation.
    **kwargs : dict
        Additional keyword arguments that may control aspects of the uncertainty simulation, such as the distribution of simulated noise, seed for reproducibility, among others.

    Returns
    -------
    uframe
        A `uframe` object containing the data with simulated uncertainty. This object is ready for analysis, allowing for the exploration of uncertainty's impact on data interpretation and decision-making processes.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> uf_sim = uframe_from_array_sim(X, p=0.3)

    # Example: Using the resulting uframe for analysis
    >>> print(uf_sim.sample(100))  # Generate 100 samples for each data instance to analyze the simulated uncertainty
    >>> print(uf_sim.ev())         # Calculate expected values to understand the central tendency of the simulated uncertain data

    Notes
    -----
    - The specific mechanism of uncertainty simulation (e.g., noise addition, value replacement) should be detailed in the implementation to guide users on the nature of simulated uncertainty within the data.
    - The `p` parameter's interpretation and the effect of additional keyword arguments (`**kwargs`) might vary based on the implementation specifics, highlighting the flexibility of this function in simulating various uncertainty scenarios.
    """

    X_missing, missing = generate_missing_values(
        complete_data=X, p=p, seed=seed, method=missing_method)

    if std_method == "relative":
        stds = std * np.std(X, axis=1)
    else:
        stds = np.repeat(std, X.shape[1])

    distr = {}
    for i in range(X.shape[0]):
        if np.any(np.isnan(X_missing[i, :])) == False:
            continue

        mean = add_bias(X[i, :][np.where(np.isnan(X_missing[i, :]))[0]],
                        bias=bias,
                        bias_method=bias_method)

        if dist_method == "gaussian":
            cov = np.diag(stds[np.where(np.isnan(X_missing[i, :]))])

            distr[i] = stats.multivariate_normal(mean=mean, cov=cov)

    u = uframe()
    u.append(new=[X_missing, distr])

    return u


def uframe_from_array_mice(a: np.ndarray, p=0.1,
                           mice_iterations=5,
                           kernel="gaussian",
                           method='binomial',
                           cat_indices=[],
                           seed=None,
                           **kwargs):
    """
    Create a uframe object from a numpy array with missing data imputed using the MICE algorithm.

    This function applies MICE (Multiple Imputation by Chained Equations) to impute missing values in the given array, then encapsulates the imputed data along with imputation distributions for each missing value into a `uframe` object.

    Parameters
    ----------
    a : np.ndarray
        The input data array with missing values to be imputed. Should be a 2D numpy array.
    p : float, default=0.1
        The proportion of missingness if the input array `a` does not already contain missing values.
    mice_iterations : int, default=5
        The number of iterations the MICE algorithm will run to impute missing values.
    kernel : str, default="gaussian"
        The kernel to use for density estimation of continuous variables. Supported values include "stats.gaussian_kde" for scipy Gaussian kernels. and "gaussian" for sklearn gaussian kernel.
    method : str, default='binomial'
        The method to generate missing values artificially. Possible values are "binomial", "attributes" and "instances". For binomial, each selection is treated as an independant event with a selection probability of p. For "attributes", p percentage of entries in each attribute/column are selected. For "instances", p percentage of entries in each instance/row are selected. 
    cat_indices : list, default=[]
        List of column indices in `a` that are categorical. These columns will be treated differently during the imputation process.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
    **kwargs : dict
        Additional keyword arguments for the MICE imputation function.

    Returns
    -------
    uframe
        A `uframe` object containing the imputed data along with imputation distributions for each missing entry, ready for further analysis.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
    >>> uf = uframe_from_array_mice(a, mice_iterations=10, cat_indices=[2], seed=42)

    # Example: Using the resulting uframe for analysis
    >>> print(uf.sample(100))  # Generate 100 samples for statistical analysis
    >>> print(uf.ev())         # Calculate expected values for the uframe

    Notes
    -----
    - MICE is particularly useful for datasets where the assumption of data being missing at random (MAR) is reasonable.
    - The choice of `kernel` and the number of `mice_iterations` can significantly impact the quality of imputation.
    """
    x, missing = generate_missing_values(a, p, seed, method=method)

    distr = {}
    cat_distr = {}
    index_dict = {}

    # train mice imputation correctly
    kds = mf.ImputationKernel(
        x,
        save_all_iterations=True,
        random_state=seed)

    kds.mice(mice_iterations)
    for i in range(x.shape[0]):
        imp_distr = None
        imp_arrays = []
        cat_distributions = []
        for j in range(x.shape[1]):
            if np.isnan(x[i, j]) and j not in cat_indices:

                imp_values = []

                for k in range(mice_iterations):
                    imput_x = kds.complete_data(iteration=k)
                    imp_values.append(imput_x[i, j])

                imp_value_arr = np.array(
                    imp_values).reshape((1, mice_iterations))
                imp_arrays.append(imp_value_arr)

                if i in index_dict.keys():
                    index_dict[i][0].append(j)
                else:
                    index_dict[i] = [[j], []]

            if np.isnan(x[i, j]) and j in cat_indices:

                imp_values = []
                for k in range(mice_iterations):
                    imput_x = kds.complete_data(iteration=k)
                    imp_values.append(imput_x[i, j])

                d = {}
                for imp_value in imp_values:
                    if int(imp_value) in d.keys():
                        d[int(imp_value)] += 1/mice_iterations
                    else:
                        d[int(imp_value)] = 1/mice_iterations

                cat_distributions.append(d)

                if i in index_dict.keys():
                    index_dict[i][1].append(j)
                else:
                    index_dict[i] = [[], [j]]

            cat_distr[i] = cat_distributions

        if len(imp_arrays) == 0:
            continue

        imp_array = np.concatenate(imp_arrays, axis=0)

        if kernel == "stats.gaussian_kde":
            kde = stats.gaussian_kde(imp_array)

        else:
            imp_array = imp_array.T
            kde = KernelDensity(kernel=kernel).fit(imp_array)

        imp_distr = kde

        distr[i] = imp_distr

    cont_indices = [i for i in range(x.shape[1]) if i not in cat_indices]
    x_cont = x[:, cont_indices]
    x[:, cont_indices] = x_cont

    u = uframe()
    u.append(new=[x, distr, cat_distr, index_dict])

    return u
