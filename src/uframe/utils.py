import pickle
from .uframe_instance import uframe_instance
import uframe
import numpy as np


def load_uframe(file):
    
    with open(file, 'rb') as f:
        l = pickle.load(f)

    instances = [] 
    for i in range(len(l)):
        instances.append(uframe_instance(certain_data = l[i][0],
                                         continuous = l[i][1],
                                         categorical = l[i][2],
                                         indices = l[i][3]))

    uf = uframe()
    uf.append_from_uframe_instance(instances)

    return(uf)


def analysis_table(true, preds): 
    residues = true - preds
    true_q= np.array([sum(true[k]<true)/len(true) for k in range(len(true))])
    new_q = np.array([sum(preds[k]<true)/len(true) for k in range(len(true))])

    return [["MAE", str(round(np.mean(np.abs(residues)),2))],
            ["RMSE", str(round(np.sqrt(np.mean(residues**2)),2))],
            ["Var",str(round(np.var(residues),2))],
            ["Diff Quantile", str(round(np.mean(np.abs(true_q - new_q)),2))]]
            
