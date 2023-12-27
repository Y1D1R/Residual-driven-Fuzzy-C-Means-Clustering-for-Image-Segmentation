# Residual-driven-Fuzzy-C-Means-Clustering-for-Image-Segmentation
This project focuses on implementing the <a href="https://ieeexplore.ieee.org/document/9242330">Residual-driven Fuzzy C-Means Clustering for Image Segmentation</a> algorithm in Python. The repository provides a brief overview of the algorithm steps and dives into the implementation and the results.

It is carried out as part of the 'Modélisation des systèmes de vision' module in the Master 2 Vision et Machine Intelligente program at the University of Paris Cité.
## Overview
This project implements the Residual-driven Fuzzy C-Means (RFCM) algorithm for color image segmentation based on the work by Cong Wang, Witold Pedrycz, ZhiWu Li, and MengChu Zhou <a href="https://ieeexplore.ieee.org/document/9242330">[link]</a>. RFCM addresses the limitations of traditional Fuzzy C-Means (FCM) by incorporating a residual-related regularization term to precisely estimate noise, enhancing clustering performance.

The main idea is to integrate a residue-driven regularization term into the <a href= "https://www.tandfonline.com/doi/abs/10.1080/01969727308546046">FCM</a> algorithm to accurately estimate noise and improve clustering performance.

The authors propose a framework that integrates spatial information and introduces a weighted regularization term to handle mixed or unknown noise, to enable more accurate noise estimation and the use of a noise-free image in the clustering process for improved results.

The algorithm, called WRFCM, is designed to balance the effectiveness and efficiency of clustering and improve existing FCM variants by considering accurate noise estimation.

The effectiveness of WRFCM is demonstrated by experiments on synthetic, medical, and real images, showing superior segmentation results to those of other FCM variants.

The algorithm enables accurate estimation of residuals (noise) and can be run with low computational overhead.

Finally, this approach contributes to improving the performance of image segmentation in the presence of mixed or unknown noise.

## Algorithm
<p align="center">
  <img src="images/fig1.jpg" width='600' />
</p>

The algorithm iteratively updates membership degrees, calculates cluster centers, residual matrix, and updates weights until convergence, effectively segmenting the image.

The mathematical formulas are as follows :

* (13)
$u_{ij}(t+1) = \frac{\left(\sum_{n \in N_j} \frac{\|x_n - r_n(t) - v_i(t)\|^2}{1 + d_{nj}}\right)^{-\frac{1}{m-1}}}{\sum_{q=1}^c \left(\sum_{n \in N_j} \frac{\|x_n - r_n(t) - v_q(t)\|^2}{1 + d_{nj}}\right)^{-\frac{1}{m-1}}}$

* (14)
$v_{ij}^{(t+1)} = \frac{\sum_{j=1}^K \left( \left( u_{ij}^{(t+1)} \right)^m \sum_{n \in N_j} \frac{x_n - r_n^{(t)}}{1 + d_{nj}} \right)}{\sum_{j=1}^K \left( \left( u_{ij}^{(t+1)} \right)^m \sum_{n \in N_j} \frac{1}{1 + d_{nj}} \right)}$


* (7)
$w_{jl} = e^{-\xi r^2_{jl}}$

## Parameters
* U : Fuzzy membership matrix where $u_{ij}$ represents the degree of membership of data point $x_j$ to cluster i.

* V : Matrix containing the cluster centroids $v_i$

* R : residual (noise)

* W : weight matrix

* β : parameter controls the impact of the fidelity term

* ||.|| : Euclidiean distance

* K : Number of points

* C : Number of Clusters
  
* m : parameter to control the fuzziness of the clustering

* n : local window of size

* $\xi$ : a positive parameter, which aims to control the decreasing rate of W


## Key Features
* **Residual-driven Approach :**  Integration of a residual-related fidelity term for accurate noise estimation.

* **Spatial Information Integration :** Framework incorporating spatial information for improved segmentation.

* **Weighted Regularization Term:** Introduction of a weighted -norm regularization term to handle mixed or unknown noise.

* **Low Computational Overhead:** Efficient execution with minimal computational burden.

* **Enhanced Segmentation in Noisy Environments:** Contribution to improved image segmentation in the presence of mixed or unknown noise.

## Usage
Clone the repository: 
```
$ git clone https://github.com/Y1D1R/Residual-driven-Fuzzy-C-Means-Clustering-for-Image-Segmentation.git
``` 

Run the main script: 
``` 
$ python main.py 
```

## Results
[Include visual results or link to a separate document showcasing segmentation results]

## Citation
If you use this code, please cite the original paper:

Cong Wang, Witold Pedrycz, ZhiWu Li & MengChu Zhou. “Residual-driven Fuzzy C-Means Clustering for Image Segmentation”. en. In : IEEE/CAA JOURNAL OF AUTOMATICA SINICA. 8 (2021), p. 876-889.

## Contribution
Feel free to contribute by opening issues or submitting pull requests.
