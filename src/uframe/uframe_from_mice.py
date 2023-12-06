import numpy as np 
from sklearn.preprocessing import  MinMaxScaler
import miceforest as mf
from scipy import stats
from sklearn.neighbors import KernelDensity
from uframe import uframe
import random

def generate_missing_values(complete_data, p, seed, method = 'binomial'):
    shape = complete_data.shape
    data_out = complete_data.copy()
    np.random.seed(seed)
    random.seed(seed)
    if p == 0: 
        missing = None
        
    elif method == 'binomial': 
        missing = np.random.binomial(1, p, shape)
      
    elif method == 'fix': 
        
        n_choose = round(p*shape[0])
        missing=np.zeros(shape)
        for i in range(shape[1]):
            missing[random.sample(range(shape[0]),n_choose),i] = 1
            
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
            new_values[i] = new_values[i] + random.uniform(-abs(bias), abs(bias))
        
    
    return new_values
    
    
def uframe_from_array_sim(X: np.ndarray, p=0.5, 
                          missing_method = 'binomial',
                          bias = 0.5,
                          bias_method = 'even',
                          dist_method = 'gaussian',
                          std  = 0.1,
                          std_method = 'fix',
                          seed  = None):  
    
    
    X_missing, missing = generate_missing_values(complete_data = X, p = p, seed = seed , method = missing_method)
    
    
    if std_method == "relative":
        stds = std * np.std(X, axis=1)
    else:
        stds = np.repeat(std, X.shape[1])


    distr = {}
    for i in range(X.shape[0]):
        if np.any(np.isnan(X_missing[i, :])) == False:
            continue
        

        
        mean = add_bias(X[i, :][np.where(np.isnan(X_missing[i, :]))[0]], 
                        bias = bias, 
                        bias_method = bias_method)
        
        
        if dist_method == "gaussian": 
            cov = np.diag(stds[np.where(np.isnan(X_missing[i, :]))])

            distr[i] = stats.multivariate_normal(mean=mean, cov=cov)


    
    u = uframe()
    u.append(new=[X_missing, distr])
    
    return u 
    
    

def uframe_from_array_mice(a: np.ndarray, p=0.1, 
                             mice_iterations = 5, 
                             kernel="gaussian",
                             method= 'binomial',
                             cat_indices=[], 
                             seed = None, 
                             **kwargs):

    x, missing = generate_missing_values(a, p, seed, method = method)
    
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
                    if int(imp_value) in d.keys():
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
            kde = KernelDensity(kernel=kernel).fit(imp_array)

        imp_distr = kde

        distr[i] = imp_distr

    cont_indices = [i for i in range(x.shape[1]) if i not in cat_indices]
    x_cont = x[:,cont_indices]
    x[:,cont_indices]=x_cont

    u = uframe()
    u.append(new=[x, distr, cat_distr, index_dict])

    return u

