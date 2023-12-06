import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA

def analysis_table(true, preds): 
    residues = true - preds
    true_q= np.array([sum(true[k]<true)/len(true) for k in range(len(true))])
    new_q = np.array([sum(preds[k]<true)/len(true) for k in range(len(true))])

    return [["MAE", str(round(np.mean(np.abs(residues)),2))],
            ["RMSE", str(round(np.sqrt(np.mean(residues**2)),2))],
            ["Var",str(round(np.var(residues),2))],
            ["Diff Quantile", str(round(np.mean(np.abs(true_q - new_q)),2))]]

def analysis_table_distr(values, other_dist,KL=""): 
    
    
    dist1 = values
    dist2= other_dist
    
    bins = np.histogram_bin_edges(np.concatenate((dist1, dist2)), bins=100)
    hist1, _ = np.histogram(dist1, bins=bins, density=True)
    hist2, _ = np.histogram(dist2, bins=bins, density=True)

    epsilon = 1e-10
    hist1 += epsilon
    hist2 += epsilon

    kl_divergence = entropy(hist1, hist2, base=np.e)

    return [["Mean",str(round(np.mean(values),2))],
            ["Var",str(round(np.var(values),2))],
            ["KL"+KL,str(round(kl_divergence,2))]]


def KL(dist1,dist2): 
    bins = np.histogram_bin_edges(np.concatenate((dist1, dist2)), bins=100)
    hist1, _ = np.histogram(dist1, bins=bins, density=True)
    hist2, _ = np.histogram(dist2, bins=bins, density=True)

    epsilon = 1e-10
    hist1 += epsilon
    hist2 += epsilon

    kl_divergence = entropy(hist1, hist2, base=np.e)

    return round(kl_divergence,2)
    


def plot_pca(X,axs, uncertain):
    # Standardize the data
    # Initialize PCA and fit the standardized data
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)

    # Separate the principal components by certainty
    certain_points = principal_components[~uncertain]
    uncertain_points = principal_components[uncertain]

    # Plot the certain points in blue and uncertain points in red
    axs.scatter(certain_points[:, 0], certain_points[:, 1], c='blue', label='Certain')
    axs.scatter(uncertain_points[:, 0], uncertain_points[:, 1], c='red', label='Uncertain')
    
    # Add labels and title
    axs.set_xlabel('Component 1')
    axs.set_ylabel('Component 2')
    axs.set_title('2 component PCA with true_data')
    
    # Add a legend to the plot
    axs.legend()