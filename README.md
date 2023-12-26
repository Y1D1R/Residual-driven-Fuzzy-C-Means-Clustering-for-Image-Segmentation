# Residual-driven-Fuzzy-C-Means-Clustering-for-Image-Segmentation
This project focuses on implementing the <a href="https://ieeexplore.ieee.org/document/9242330">Residual-driven Fuzzy C-Means Clustering for Image Segmentation</a> algorithm in Python. The repository provides a brief overview of the algorithm steps and dives into the implementation and the results.
## Overview
This project implements the Residual-driven Fuzzy C-Means (RFCM) algorithm for color image segmentation based on the work by Cong Wang, Witold Pedrycz, ZhiWu Li, and MengChu Zhou <a href="https://ieeexplore.ieee.org/document/9242330">[link]</a>. RFCM addresses the limitations of traditional Fuzzy C-Means (FCM) by incorporating a residual-related regularization term to precisely estimate noise, enhancing clustering performance.

## Key Features
Residual-driven Approach: Integration of a residual-related fidelity term for accurate noise estimation.

Spatial Information Integration: Framework incorporating spatial information for improved segmentation.

Weighted Regularization Term: Introduction of a weighted -norm regularization term to handle mixed or unknown noise.

WRFCM Algorithm: The proposed algorithm, WRFCM, balances clustering efficiency and effectiveness by considering precise noise estimation.

Demonstrated Efficacy: Experimental validation on synthetic, medical, and real images showcasing superior segmentation results compared to other FCM variants.

Low Computational Overhead: Efficient execution with minimal computational burden.

Enhanced Segmentation in Noisy Environments: Contribution to improved image segmentation in the presence of mixed or unknown noise.

## Usage
Clone the repository: 
```git clone [repository_url]``` 

Install dependencies: [List dependencies and versions] 

Run the main script: ```python main.py```

## Results
[Include visual results or link to a separate document showcasing segmentation results]

## Citation
If you use this code, please cite the original paper:

Cong Wang, Witold Pedrycz, ZhiWu Li & MengChu Zhou. “Residual-driven Fuzzy C-Means Clustering for Image Segmentation”. en. In : IEEE/CAA JOURNAL OF AUTOMATICA SINICA. 8 (2021), p. 876-889.

## Contribution
Feel free to contribute by opening issues or submitting pull requests.
