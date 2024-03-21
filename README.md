# uframe: Python uncertain data handling toolkit
[![Tests](https://github.com/URWI2/uframe/actions/workflows/tests.yml/badge.svg)](https://github.com/URWI2/uframe/actions/workflows/tests.yml)
[![PyPI Latest Release](https://img.shields.io/pypi/v/uframe.svg)](https://pypi.org/project/uframe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is it?

Introducing **uframe**: A Flexible and Powerful Framework for Handling Uncertain Data

As of today, uframe is still in development. It is a working prototype and planned features, bugfixes, tests and documentation are partly still missing. 

**uframe** is a comprehensive Python package designed to simplify and expedite research and workflows involving uncertain data from various sources.

## Table of Contents

- [Installation](#Installation)
- [Getting Started](#Getting-started)
- [Dependencies](#dependencies)
- [License](#license)
- [Documentation](#documentation)


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
- Kernel Density Estimations 
- Parametric probability distributions

```sh
#create list of uncertain instances
uncertain = []
for i in range(10): 
    uncertain.append(gaussian_kde(np.random.uniform(low = i/2, high = i+1, size= 100)))

#create uframe object 
data = uf.uframe(uncertain)
```

Once the uframe is created, a number of functions are available to the user. 

```sh
# samples two instance from the whole data set
data.sample(2)

# Calculating the mode of each instance using an optimization algorithm
data.mode()

# Determining the expected value of each instance:
data.ev()

```

uframe offers a comprehensive analysis tool. 
```sh
#Creates an in-depth analysis for data and saves it as a pdf file "test_data.pdf"
data.analyze("test_data") 
```



## Dependencies
- [NumPy - Enables working with large, multi-dimensional arrays, matrices and provides high-level mathematical functions to operate on these arrays](https://www.numpy.org)
- [miceforest - Fast, memory efficient Multiple Imputation by Chained Equations (MICE) with lightgbm, that can be used to create uncertain data](https://pypi.org/project/miceforest/)

## License
[MIT](LICENSE)

## Documentation
The official documentation is not yet available. 

[Go to Top](#table-of-contents)
