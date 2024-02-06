Examples
=========

Here are some exmaples to get you started with the uframe package. 


Example 1: Introduciton example 
-------------------------------------------------------

-- code-block:: python

    #import packages
    import uframe as uf 
    import numpy as np
    from scipy.stats import gaussian_kde

    #creating uncertain data instances using scipy KDEs over random Data
    uncertain = []
for i in range(10): 
    uncertain.append(gaussian_kde(np.random.uniform(low = i/2, high = i+1, size= 100)))

    data = uf.uframe()
    data.append(uncertain)


    # samples one instance from the whole data set
    data.sample()

    # Calculating the mode of each instance using an optimization algorithm
    data.mode()

    # Determining the expected value of each instance:
    data.ev()

Example 2: Simulating a uncertain data  
-------------------------------------------------------

-- code-block::python

    import uframe as uf 
    from sklearn.datasets import load_iris
    import numpy as np 
 
    #Preparing data
    data = load_iris() 
    X = data.data
    y = data.target 
 
    #delete specific entries  
    num_del= 100   
    rows, cols = X.shape
    row_indices = np.random.choice(rows, num_del)
    col_indices = np.random.choice(cols, num_del)

    X[row_indices, col_indices] = np.nan
 
    # Generate multiple predictions for each missing value and fit a KDE over these predictions. This result is then saved as a uframe
    uncertain = uf.uframe_from_array_mice(X, p=0)
 	
	
    # delete 50% of data entries to generate uframe 
    X = data.data
    uncertain = uf.uframe_from_array_mice(X, p=.5, seed = 42)

    #again the standard methods can be applied to the resulting uframe: 
   # samples one instance from the whole data set
    data.sample()

    # Calculating the mode of each instance using an optimization algorithm
    data.mode()

    # Determining the expected value of each instance:
    data.ev()



