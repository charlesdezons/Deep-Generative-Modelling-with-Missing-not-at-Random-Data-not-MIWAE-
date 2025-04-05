# Deep Generative Modelling with Missing Not At Random Data (not-MIWAE)

This repository contains the implementation and experiments for the project "Deep Generative Modelling with Missing Not At Random Data (not-MIWAE)" as part of the MVA course "Introduction to Probabilistic Graphical Models and Deep Generative Models." The project focuses on the paper by Ipsen, Mattei, and Frellsen (2021) and explores deep generative modeling techniques for handling missing data that is Missing Not At Random (MNAR).

## Table of Contents

- [Project Context](#project-context)
- [Key Contributions](#key-contributions)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

## Project Context

This project aims to address the challenges of inferring and imputing missing data, particularly when the data is Missing Not At Random (MNAR). Unlike Missing At Random (MAR) or Missing Completely At Random (MCAR) scenarios, MNAR data requires more sophisticated modeling as the missingness mechanism depends on the missing values themselves.

## Key Contributions

- **Deep Latent Variable Models (DLVMs)**: Utilized for inference and imputation in missing data problems, particularly effective for high-dimensional data.
- **not-MIWAE**: An extension of the Importance Weighted AutoEncoder (IWAE) framework tailored for MNAR data, inspired by the MIWAE framework for MAR scenarios.

## Methodology

### Notations and Hypotheses

- **Dataset**: $\mathbf{X} = (x_1, \ldots, x_n) \in \mathcal{X}^n$
- **Missingness Mask**: $s \in \{0,1\}^p$
- **Assumptions**: MCAR, MAR, MNAR

The joint distribution $p_{\theta, \phi}(x, s)$ is factorized as $p_{\theta, \phi}(x, s) = p_{\theta}(x) p_{\phi}(s \mid x)$, with different assumptions for MCAR, MAR, and MNAR.

### Model Architecture

The not-MIWAE framework extends the VAE and IWAE architectures to handle MNAR data by optimizing both data-generation and missingness mechanisms. The objective function is estimated using importance sampling and Monte Carlo sampling.

### Imputation

Once the model is trained, missing values are imputed using the squared error as a loss function. The optimal imputations minimize the expected loss given the observed data and the missingness mask.

## Experiments

### Synthetic Data

- **Gaussian Distribution**: 2D Gaussian data with a specific missingness mask.
- **Ring-shaped Distribution**: 2D ring data with angular sector-based missingness.

### Real Dataset

- **MNIST Dataset**: Experiments with a circular mask applied to MNIST images to simulate missing data.

## Results

### Reconstruction Quality

- **Gaussian Data**: Effective reconstruction using available information.
- **Ring Data**: Challenges in capturing the non-linear structure.
- **MNIST Data**: Visual and quantitative analysis showed limitations in reconstructing occluded regions.

### RMSE Performance

- **not-MIWAE**: Consistently achieved the lowest RMSE across datasets.
- **Comparison**: Outperformed heuristic methods like KNN and Random Forest.

### Inferred Distributions

- **Gaussian Dataset**: Effectively captured the missing data dynamics.
- **Ring Dataset**: Limited by the linear approximation of the missingness mechanism.

## Usage

### Prerequisites

- Python 3.x
- Libraries: NumPy, PyTorch, Matplotlib, Scikit-learn, Weights & Biases

### Installation

```bash
git clone https://github.com/yourusername/not-MIWAE.git
cd not-MIWAE
pip install -r requirements.txt
