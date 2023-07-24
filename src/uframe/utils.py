import pickle

def load_uframe(file):
    
    with open(file, 'rb') as f:
        d = pickle.load(f)
    return(d)
