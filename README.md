# uframe: Python uncertain data handling toolkit
[![Tests](https://github.com/URWI2/uframe/actions/workflows/tests.yml/badge.svg)](https://github.com/URWI2/uframe/actions/workflows/tests.yml)
[![PyPI Latest Release](https://img.shields.io/pypi/v/uframe.svg)](https://pypi.org/project/uframe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/URWI2/uframe/branch/main/graph/badge.svg?token=BQJ6FYZIWR)](https://codecov.io/gh/URWI2/uframe)

## What is it?

Introducing **uframe**: A Flexible and Powerful Framework for Handling Uncertain Data

**uframe** is a comprehensive Python package designed to simplify and expedite research involving uncertain data from various sources.
By seamlessly integrating Probability Density Functions (PDFs) into its core functionality, uframe empowers users to manipulate and analyze uncertain data with ease.

Key Features:

- Flexible Data Handling: uframe offers a versatile framework for working with uncertain data, enabling researchers to tackle complex datasets efficiently.
- Probability Density Functions (PDFs): uframe leverages PDFs as the fundamental representation of attribute uncertainty, allowing for a comprehensive characterization of uncertainty within each instance.
- Streamlined Research: With its user-friendly interface and powerful capabilities, uframe facilitates seamless data exploration, analysis, and visualization, promoting accelerated research and insights.

Whether you're dealing with incomplete or imprecise measurements, or simply need a reliable tool to handle uncertainty, uframe provides a robust foundation for your uncertain data analysis needs.

## Table of Contents

- [Main Features](#main-features)
- [Installation](#Installation)
- [Getting Started](#Getting-started)
- [Dependencies](#dependencies)
- [License](#license)
- [Documentation](#documentation)

## Main Features


## Installation
The source code is available on GitHub:
https://github.com/URWI2/uframe

Binary installers of the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/uframe) .


```sh
pip install uframe
```


## Getting started


To begin working with uncertain data using uframe, follow these steps:

1. Import the necessary libraries:
```sh
import uframe as uf 
import numpy as np
from scipy.stats import gaussian_kde
```
2. Representing uncertain data
Uncertain data can be represented by different functions: 
- Kernel Density Estimations (eiter scipy or sklearn)
- Scipy probability distributions 

```sh
uncertain = []
for i in range(10): 
    uncertain.append(gaussian_kde(np.random.uniform(low = i/2, high = i+1, size= 100)))

data = uf.uframe()
data.append(uncertain)
```

Once the uframe is created, a number of functions are available to the user. 

```sh
# samples one instance from the whole data set
data.sample()

# Calculating the mode of each instance using an optimization algorithm
data.mode()

# Determining the expected value of each instance:
data.ev()

```




### uframe_instance subclass 
uframe is built upon instances of the class uframe_instance. These classes can also be created by the user and combined to a uframe. 
The Argument indices is necessary if the order differs of certain and uncertain attributes differs between instances. 
Indices expects a list of three lists, with the indices of certain, continuous and categorical attributes in different lists.  
```sh
import uframe as uf
from scipy.stats import gamma, norm
uncertain = [uf.uframe_instance(certain_data = np.array([2.1]), continuous = [norm(0.2,1), gamma(0.3)], indices = [[1],[0,2],[]]),
             uf.uframe_instance(continuous = [norm(0.1,1), norm(0.3,0.7), gamma(1)], indices = [[],[1,0,2],[]])]
data = uf.uframe()
data.append(uncertain)

data.sample()
```




## Dependencies
- [NumPy - Enables working with large, multi-dimensional arrays, matrices and provides high-level mathematical functions to operate on these arrays](https://www.numpy.org)


## License
[MIT](LICENSE)

## Documentation
The official documentation is not yet available. 

[Go to Top](#table-of-contents)
