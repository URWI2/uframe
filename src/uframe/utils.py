import pickle
from .uframe_instance import uframe_instance
from .uframe import uframe as ufr



def load_uframe(file):
    
    with open(file, 'rb') as f:
        l = pickle.load(f)

    instances = [] 
    for i in range(len(l)):
        instances.append(uframe_instance(certain_data = l[i][0],
                                         continuous = l[i][1],
                                         categorical = l[i][2],
                                         indices = l[i][3]))

    uf = ufr()
    uf.append_from_uframe_instance(instances)

    return(uf)


