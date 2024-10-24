from scipy.stats import norm
import numpy as np

def certification(K, K_b, n_b, epsilon, H, beta, alpha):

    tau = 1.0 * (K-K_b) / K_b
    term = H * norm.ppf(1 - beta/(2*H*K_b)) / 2.0 / np.sqrt(n_b) * (1 + 2.0 * tau  / (1 - tau))
    lower_bnd = 1 - alpha - (epsilon * n_b + 1) / (n_b + K_b) - term
    upper_bnd = 1 - alpha + epsilon + K_b / (n_b + K_b) + term
    # print(f'{tau}:{term}')
    # print(tau)
    print(H*norm.ppf(1 - beta/(2*H*K_b)) / 2.0 / np.sqrt(n_b) )
    print(term)
    if upper_bnd>1.0:
        upper_bnd=1.0
    if lower_bnd<0.0:
        lower_bnd=0.0
    return lower_bnd, upper_bnd

# print( norm.ppf(1 - 1e-1/(2*100*10)))

# size = 100000
# K = 9
# K_b = 5
# print(certification(K,K_b,40000,0,H=2,beta=0.05,alpha=0.1))