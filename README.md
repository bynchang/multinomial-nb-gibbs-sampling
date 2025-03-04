# Gibbs Sampling for a Naive Bayes Classifier with Multinomial Likelihood 
This repository extends the methodology presented in [Gibbs Sampling for the Uninitiated](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=fc024fbdc59c3b5e708268b29e00cebaf9593875) (Resnik and Hardisty, 2010) to handle multi-class classification problems.

## Modifications

### Model Parameters
- Labels $L_j \in \{0, 1, 2, ..., K-1\}$ for $K$ classes (rather than binary $\{0, 1\}$)
- Class probabilities $\pi = (\pi_0, \pi_1, ..., \pi_{K-1})$ following a Dirichlet distribution (generalizing Beta)
- Word distributions $\theta_0, \theta_1, ..., \theta_{K-1}$ for each class

### Sampling Equations

For document labels, the sampling probability becomes:

$\text{Pr}(L_j = x|L^{(-j)}, C^{(-j)}, \theta_0, \theta_1, \ldots, \theta_{K-1}; \mu) \propto \frac{C_x + \gamma_{\pi x} - 1}{N + \sum_{k=0}^{K-1} \gamma_{\pi k} - 1} \prod_{i=1}^V \theta_{x,i}^{W_{ji}}$

Where:
- $C_x$ is the number of documents with label $x$
- $γ_π$ are the Dirichlet hyperparameters for class probabilities
- $θ_{x,i}$ is the probability of word $i$ in class $x$
- $W_{ji}$ is the frequency of word $i$ in document $j$
