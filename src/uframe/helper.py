import numpy as np
from scipy.stats import entropy

def analysis_table(true, preds): 
    residues = true - preds
    true_q= np.array([sum(true[k]<true)/len(true) for k in range(len(true))])
    new_q = np.array([sum(preds[k]<true)/len(true) for k in range(len(true))])

    return [["MAE", str(round(np.mean(np.abs(residues)),2))],
            ["RMSE", str(round(np.sqrt(np.mean(residues**2)),2))],
            ["Var",str(round(np.var(residues),2))],
            ["Diff Quantile", str(round(np.mean(np.abs(true_q - new_q)),2))]]
            


def analysis_table_distr(values, other_dist): 
    
    
    dist1 = values
    dist2= other_dist
    
    bins = np.histogram_bin_edges(np.concatenate((dist1, dist2)), bins='auto')
    hist1, _ = np.histogram(dist1, bins=bins, density=True)
    hist2, _ = np.histogram(dist2, bins=bins, density=True)

    epsilon = 1e-10
    hist1 += epsilon
    hist2 += epsilon

    kl_divergence = entropy(hist1, hist2, base=np.e)
        
    
    return [["Mean",str(round(np.mean(values),2))],
            ["Var",str(round(np.var(values),2))],
            ["KL",str(round(kl_divergence,2))]]