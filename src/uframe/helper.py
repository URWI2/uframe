import numpy as np

def analysis_table(true, preds): 
    residues = true - preds
    true_q= np.array([sum(true[k]<true)/len(true) for k in range(len(true))])
    new_q = np.array([sum(preds[k]<true)/len(true) for k in range(len(true))])

    return [["MAE", str(round(np.mean(np.abs(residues)),2))],
            ["RMSE", str(round(np.sqrt(np.mean(residues**2)),2))],
            ["Var",str(round(np.var(residues),2))],
            ["Diff Quantile", str(round(np.mean(np.abs(true_q - new_q)),2))]]
            
