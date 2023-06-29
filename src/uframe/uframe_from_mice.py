import numpy as np 
from sklearn.preprocessing import  MinMaxScaler
import miceforest as mf
from scipy import stats
from sklearn.neighbors import KernelDensity
from uframe import uframe

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
                #diese Schleife durch list comprehension ersetzen 
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

    
    scaler= MinMaxScaler()
    x = scaler.fit_transform(x)
    #a[missing==1]=np.nan
    u = uframe()
    u.append(new=[x, distr])
    

    return u



def generate_missing_values(complete_data, p):
    shape = complete_data.shape
    y = complete_data.copy()
    missing = np.random.binomial(1, p, shape)
    
    y[missing.astype('bool')] = np.nan
    return y, missing


def uframe_from_array_mice_2(a: np.ndarray, p=0.1, mice_iterations=5, kernel="stats.gaussian_kde",
                           scaler= "min_max", cat_indices=[]):

    x, missing = generate_missing_values(a, p)

    distr = {}
    cat_distr = {}
    index_dict={}

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

    #print(x)
    cont_indices = [i for i in range(x.shape[1]) if i not in cat_indices]
    #print("Cont indices")
    scaler= MinMaxScaler()
    x_cont = scaler.fit_transform(x[:,cont_indices])
    #print("X cont scaled", x_cont)
    x[:,cont_indices]=x_cont
    #print("X after scaling", x, x[:,cat_indices])
    #a[missing==1]=np.nan
    u = uframe()
    
    #print("Cat distr", cat_distr)
    u.append(new=[x, distr, cat_distr, index_dict])

    #print("Col dtype", u._col_dtype)
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

    print(x, missing)
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
