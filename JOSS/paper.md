---
title: 'uframe: A Python package for uncertain dataframes'
tags:
- Python
- data
- unertainty
- uncertain

authors:
	- name: Christian Amesoeder
	  orcid: 0000-0002-1668-8351
	  equal-contrib: true
	  affiliation: 1 
	- name: Michael Hagn
	  equal-contrib: true 
	  affiliation: 1 
	  affiliations:

affiliations:
	- name: WI2 University of Regensburg, Germany
	  index: 1
date: 21 March 2024
bibliography: paper.bib

---

# Summary

'uframe' is an open-source Python package designed to simplify working with uncertain data from various sources, streamlining the integration of uncertainty into the data analysis process. 
'uframe' allows data instances to be defined by a mixture of certain and uncertain attribute values, where uncertain attributes are further categorized into continuous and categorical types. Continuous attributes are represented through either conditional parametric probability distributions or kernel density estimations. 


# Statement of need

Aleatoric uncertainty is a pervasive aspect of real-world data, yet its integration into data analysis workflows remains conspicuously sparse [@Li]. A primary hurdle is the complexity inherent in managing uncertain data within Python-based environments. Addressing this gap, we present 'uframe', a Python package for the effective handling of uncertain data.

Historically, individuals working with uncertain data resorted to crafting bespoke solutions for representing data instances with associated probability distributions, leading to a fragmented and inefficient landscape [@GuR; @Ge]. 'uframe' unifies this landscape by offering a cohesive framework for engaging with uncertain data, significantly enhancing the reproducibility, depth, and breadth of data analyses that incorporate uncertainty.


# Functionality 

The utilization of ‘uframe’ is distinguished by the instantiation of a ‘uframe’ object, followed by the application of its built-in methods. These methods enable users to access statistical measures such as mode or expected value for each data instance, evaluate the probability density function at various points, and draw samples from all instances. This functionality is pivotal for a broad spectrum of analytical applications, from probabilistic modeling to sensitivity analysis. Probability distributions can be either parametric and represented using the scipy package or kernel density estimations [@scipy; @scikit-learn]. 


# Data Simulation 

Due to the sparsity of data with known uncertainties, the simulation of uncertain datasets for research purposes is common practice. In some cases, certain attributes are replaced with normal distributions of a given variance where the mean is given by the original value [@GuR; @Ren]. In other cases, certain attributes are replaced with an equal distribution over a range of values [@Qin]. 

In the ‘uframe’ package, we introduce a more advanced approach to artificially add uncertainty in deterministic data. With the method ’uframe_from_array_mice’ uncertain data with arbitrary distributions can be generated, complete with a traceable generation process. This is achieved by omitting a fixed percentage of attribute values across a certain data set. Afterwards multiple imputations are generated for each missing attribute, for which a kernel density estimation is fitted. This approach leads to data sets with known ground truths, making them invaluable for developing and evaluating a vast range of applications, from the training of machine learning models to the validation of analytical methods.


# Example Application 

For any deterministic dataset, data uncertainty can be added in a controllable way using uframe.uframe_from_array_mice. In the application of this function, a user-determined percentage of all feature values in the dataset is deleted. For each deleted value, multiple values are imputed using Multiple Imputation by Chained Equations (MICE) before a probability distribution is fitted over these imputed values. The probability distribution is obtained by Kernel Density Estimation over the imputed values for continuous features and by determining relative frequencies of imputed values for categorical features. As the uncertain continuous distributions are modelled using a multidimensional probability distributions, possible dependencies between the features are adequately captured. The result is an uncertain dataset in which uncertainty is represented by probability distributions which allows for a mathematically sound analysis. In addition, the knowledge of the original values for all deleted features provides a baseline for further analysis. 
A uframe.analysis function allows for calculation and visualization of value distributions and uncertainty in the data, thus in particular showing the impact of feature value deletion and multiple imputation.
 
![Excerpt of the 'uframe' built-in analysis functionality showing the smoothed distribution of the variable sepal length of the Iris Dataset for the original values, the modal values and the expected values [@Iris].\label{fig:one}](example.png){ width=80% }

In order to use uncertain data in additional tasks such as the training or evaluation of machine learning models, uncertainty has to be addressed to fit the requirements of the method. Standard procedures such as mean or mode imputation of uncertain values as well as creating a deterministic dataset from sampling are readily available in uframe. Access to statistical properties of the probability distributions also facilitates the design and implementation of more sophisticated methods of handling data uncertainty in applications. 



# Acknowledgements

We acknowledge the valuable input from Lucas Luttner and Thomas Krapf.
This work has been supported by the German Research Foundation
(DFG: http://www.dfg.de) under grant scheme 494840328. 

# References
