import numpy as np

gallery_label = np.array([1,2,3,4,5,6,7,8])
def vote_n_max_decision(cos_similarity,n):
    best_match = np.argsort(cos_similarity)[::-1][0:n]
    print(best_match)
    best_match = gallery_label[best_match]
    print(best_match)

a = np.array([3,0,4,3,6,2,3,5])
vote_n_max_decision(a,5)
a.sort()
print(a[::-1][0:5])