library(MCMCpack) # For rdirichlet

# Parameters
D <- 4             # number of buckets
N <- 100           # number of observations
K <- 3             # number of clusters

# Generate synthetic data
n <- round(runif(N, 20, 200))   # sample sizes for each observations
true_pi <- c(0.2, 0.5, 0.3)     # true cluster proportions
true_z <- sample(K, N, TRUE, true_pi)
true_theta <- list(c(0.5, 0.3, 0.2, 0.1), 
                   c(0.1, 0.7, 0.1, 0.1), 
                   c(0.2, 0.1, 0.1, 0.6))

# Generate observed data x
x <- t(mapply(function(size, probs) {
  rmultinom(1, size=size, prob=probs)
}, n, true_theta[true_z]))

# Function to compute posterior probability for cluster assignment
compute_posterior_z <- function(theta, j, x, alpha_pi, cluster_counts){
  log_p <- sapply(1:K, function(i) {
    # ifelse to avoid 0 probability
    if (cluster_counts[i] + alpha_pi[i] <= 1){
      numerator <- cluster_counts[i] + alpha_pi[i]
    } else {
      numerator <- cluster_counts[i] + alpha_pi[i] - 1
    }
    log_prior <- log(numerator/(N - 1 + sum(alpha_pi)))
    log_likelihood <- sum(x[j,] * log(theta[i,] + 1e-20))
    log_prior + log_likelihood
  })
  
  # log-sum-exp trick for numerical stability
  max_log_p <- max(log_p)
  p <- exp(log_p - max_log_p)
  p <- p / sum(p)
  return(p)
}

# Hyperparameters
alpha_pi <- rep(1, K)        # hyperparameter for Dirichlet prior on proportions
alpha_theta <- rep(1, D)     # hyperparameter for Dirichlet prior on bucket probabilities

# Initialization
pi <- rdirichlet(1, alpha_pi)                # length K, cluster proportions
z <- sample(K, size=N, replace=TRUE, prob=pi) # length N, cluster assignments
theta <- matrix(0, K, D)
for (i in 1:K) {
  theta[i,] <- rdirichlet(1, alpha_theta)   # K×D matrix of bucket probabilities
}

# Initialize counters
counts_by_bucket <- matrix(0, K, D)        # sufficient statistics
for (i in 1:K) {
  counts_by_bucket[i,] <- colSums(x[z == i,, drop=FALSE])
}
cluster_counts <- tabulate(z, K)           # count of observations in each cluster

# Number of iterations
n_iter <- 30

# Store samples after burn-in
theta_samples <- array(0, dim=c(n_iter, K, D))
z_samples <- matrix(0, n_iter, N)
cluster_count_samples <- matrix(0, n_iter, K)
sample_idx <- 1

# Run Gibbs sampling
for(t in 1:n_iter){
  # Sample cluster assignments z
  for(j in 1:N){
    current_z <- z[j]
    
    # Sample new cluster
    new_z <- sample(K, 1, prob=compute_posterior_z(theta, j, x, alpha_pi, cluster_counts))
    
    # Update counts with new assignment
    counts_by_bucket[current_z,] <- counts_by_bucket[current_z,] - x[j,]
    cluster_counts[current_z] <- cluster_counts[current_z] - 1
    counts_by_bucket[new_z,] <- counts_by_bucket[new_z,] + x[j,]
    cluster_counts[new_z] <- cluster_counts[new_z] + 1
    z[j] <- new_z
  }
  # Sample new theta parameters
  theta <- t(sapply(1:K, function(i) rdirichlet(1, counts_by_bucket[i,] + alpha_theta)))
  
  # Store samples
  theta_samples[sample_idx,,] <- theta
  z_samples[sample_idx,] <- z
  cluster_count_samples[sample_idx,] <- cluster_counts
  sample_idx <- sample_idx + 1
}

# Compare true and inferred clusters
table(true_z)
table(z)
print(cluster_count_samples)
