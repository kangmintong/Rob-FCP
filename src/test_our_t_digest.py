import numpy as np
from tdigest import TDigest

alpha = 0.1
x = np.random.normal(loc=0.5,scale=1.0,size=10000)
digest = TDigest(delta=0.01, K=25)
digest.batch_update(x)
num_communi = len(digest)
q_hat = digest.percentile(round(100*alpha))
gt = np.quantile(x,alpha)
cov = np.mean(x<q_hat)

print(f'communication cost: {num_communi}')
print(f'estimated q_hat: {q_hat}')
print(f'error of scores: {abs(gt-q_hat)}')
print(f'error of coverage (%): {abs(cov-alpha)*100}')