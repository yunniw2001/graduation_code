import numpy as np

gallery_label = np.array([1,2,3,4,5,6,7,8])
def vote_n_max_decision(cos_similarity,n):
    best_match = np.argsort(cos_similarity)[::-1][0:n]
    print(best_match)
    best_match = gallery_label[best_match]
    print(best_match)
def get_score_matrix(cosine_matrix,gallery,test_batch):
    # cosine_matrix = cosine_similarity(gallery,test_batch)
    tmp = np.argmax(cosine_matrix,axis=0)
    tmp = np.expand_dims(tmp,axis=0)
    out = np.zeros_like(cosine_matrix)
    np.put_along_axis(out,tmp,1,axis=0)
    return out
a = np.random.randint(100, size=(3, 4))
print(a)
print(get_score_matrix(a,None,None))