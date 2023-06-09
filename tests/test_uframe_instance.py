import numpy as np 
import scipy
from sklearn.neighbors import KernelDensity
from src.uframe.uframe_instance import uframe_instance
import pytest
import random

def test_check():
    assert 1 == 1
    
@pytest.mark.parametrize(
    ('continuous_obj', 'shape', 'n'),
    (
        (
            None,
            0,3
        ),
        (
            scipy.stats.gaussian_kde(np.vstack([np.random.normal(size=200),np.random.normal(scale=0.5, size=200)])),
            2, 2
        ),
        (
            KernelDensity().fit(np.vstack([np.random.normal(size=200),np.random.normal(scale=0.5, size=200)]).transpose()),
            2,10
        ),
        (
            scipy.stats.gamma(2),
            1,7
        ),
        (
            [scipy.stats.norm(10), scipy.stats.norm(2)],
            2,11
        ),
        (
            scipy.stats.multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]]),
            2,5
        )
    )
)
def test_sample_shape(continuous_obj, shape, n):
    
    instance_0 =  uframe_instance(certain_data = np.array([99,88]), continuous = continuous_obj, categorical = None)
    assert instance_0.mode().shape == (1,2+shape)
    assert instance_0.sample(1).shape == (1,2+shape)
    assert instance_0.sample(n).shape == (n,2+shape)
    
    
    instance_1 =  uframe_instance(certain_data = None, continuous = continuous_obj, categorical = None)
    assert instance_1.mode().shape == (1,shape)
    assert instance_1.sample(1).shape == (1,shape)
    assert instance_1.sample(n).shape == (n,shape)



def test_certain():
    instance_2 =  uframe_instance(certain_data = np.array([1,1]),continuous = None, categorical = None)
    assert instance_2.mode().shape == (1,2)
    assert instance_2.sample(1).shape == (1,2)
    assert instance_2.sample(5).shape == (5,2)

 
def test_indices(): 
    
    ind = [*range(10)]
    random.shuffle(ind)
    certain = np.array([*range(10)])
    instance = uframe_instance(certain_data = certain , indices = [ind,[],[]])
    
    assert (instance.mode() == instance.sample()).all()
 
@pytest.mark.parametrize(
     ('cat', 'shape'),
     (
         (
             None,
             0
         ),(
             [{"a":0.1, "b":0.9}],
             1
         ),(
             [{"a":0.1, "b":0.9}, {"a":0.6, "d":0.4}],
             2
         ),
         
     )
 )
   
def test_categorical(cat,shape):
    instance = uframe_instance(certain_data=np.array([1,1]), categorical = cat)
    
    assert instance.mode().shape == (1,2+shape)
    assert instance.sample(1).shape == (1,2+shape)
    assert instance.sample(3).shape == (3,2+shape)
    
     
     