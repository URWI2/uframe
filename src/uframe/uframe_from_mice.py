import numpy as np 
from sklearn.preprocessing import  MinMaxScaler
import miceforest as mf
from scipy import stats
from sklearn.neighbors import KernelDensity
from uframe import uframe


def generate_missing_values(complete_data, p, seed):
    shape = complete_data.shape
    y = complete_data.copy()
    np.random.seed(seed)
    
    if p > 0: 
        missing = np.random.binomial(1, p, shape)
        y[missing.astype('bool')] = np.nan
    else: 
        missing = None
        
    return y, missing


def uframe_from_array_mice(a: np.ndarray, p=0.1, 
                             mice_iterations = 5, 
                             kernel="gaussian",
                             cat_indices=[], 
                             seed = None, 
                             **kwargs):

    x, missing = generate_missing_values(a, p, seed)
    
    distr = {}
    cat_distr = {}
    index_dict={}

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

                imp_value_arr = np.array(imp_values).reshape((1, mice_iterations))
                imp_arrays.append(imp_value_arr)
                
                if i in index_dict.keys():
                    index_dict[i][0].append(j)
                else:
                    index_dict[i]=[[j], []]
                
            if np.isnan(x[i,j]) and j in cat_indices:
                
                imp_values = []
                for k in range(mice_iterations):
                    imput_x = kds.complete_data(iteration=k)
                    imp_values.append(imput_x[i,j])
                
                d={}
                for imp_value in imp_values:
                    if imp_value in d.keys():
                        d[int(imp_value)]+= 1/mice_iterations
                    else:
                        d[int(imp_value)] = 1/mice_iterations 
                        
                cat_distributions.append(d)
                
                if i in index_dict.keys():
                    index_dict[i][1].append(j)
                else:
                    index_dict[i]=[[],[j]]
                
            cat_distr[i]= cat_distributions
            
        if len(imp_arrays) == 0:
            continue
        
        imp_array = np.concatenate(imp_arrays, axis=0)
        
        if kernel == "stats.gaussian_kde":
            kde = stats.gaussian_kde(imp_array)

        else:
            imp_array = imp_array.T
            kde = KernelDensity(kernel=kernel, **kwargs).fit(imp_array)

        imp_distr = kde

        distr[i] = imp_distr

    cont_indices = [i for i in range(x.shape[1]) if i not in cat_indices]
    x_cont = x[:,cont_indices]
    x[:,cont_indices]=x_cont

    u = uframe()
    u.append(new=[x, distr, cat_distr, index_dict])

    return u


# add Gaussian noise of given std to chosen entries
# relative=True: multiply std with standard deviation of the column to get the std for a column
def uframe_noisy_array(a: np.ndarray, std=0.1, relative=False, unc_percentage=0.1):

    if relative:
        #print("Get standard deviations of each column")
        stds = std * np.std(a, axis=1)
    else:
        stds = np.repeat(std, a.shape[1])

    print(stds)

    # delete unc_percentage of all values
    x, missing = generate_missing_values(a, unc_percentage)

    # create suitable dictionary of distributions
    distr = {}
    for i in range(x.shape[0]):
        if np.any(np.isnan(x[i, :])) == False:
            continue
        mean = a[i, :][np.where(np.isnan(x[i, :]))[0]]
        cov = np.diag([stds[np.where(np.isnan(x[i, :]))[0]]])

        print(mean, cov)

        distr[i] = stats.multivariate_normal(mean=mean, cov=cov)

    # generate uframe und use append function and return it
    u = uframe()
    u.append(new=[x, distr])

    return u
