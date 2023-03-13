import numpy as np
def calculate_weight():
    accuracy = [0.98,0.733,0.77]
    accuracy = np.array(accuracy)
    # normalization
    tot = sum(accuracy)
    x = accuracy/tot
    # beta_k
    accuracy_mean = np.mean(accuracy)
    sigma = np.sqrt(np.sum(np.power(accuracy-accuracy_mean,2)))
    miu = np.fabs(1-2.5*sigma)
    beta = np.exp(-np.power(x-miu,2)/(2*(sigma**2)))/(sigma*np.sqrt(2*np.pi))
    print(beta)

calculate_weight()