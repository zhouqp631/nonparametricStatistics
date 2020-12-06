import numpy as np
#%%
from scipy.special import comb, perm
def findk(k,n=10):
    p = 0
    for i in range(k+1):
        p += comb(n,i)*0.5**n
    return p


[2*findk(k) for k in range(10)]

#%%
flow = [206,223,235,264,229,217,188,204,182,230]
np.mean(flow)
tn = (np.mean(flow)-217)/np.std(flow)*np.sqrt(10)
from scipy.stats import t
print(2*(1-t.cdf(tn,9)))
#%%
a = [36, 32, 31 ,25 ,28 ,36 ,40 ,32,
     41, 26 ,35 ,35 ,32, 87, 33, 35]
tn = (np.mean(a)-37)/np.std(a)*np.sqrt(16)
print(t.cdf(tn,15))