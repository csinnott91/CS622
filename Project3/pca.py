import numpy as np

def compute_Z(X, centering=True, scaling=False):
    
    if centering==True:
        mean = np.mean(X, axis=0)
        
        for j in range(len(mean)):
            for i in range(len(X)):
                X[i, j] = X[i, j] - mean[j]
    else:
        X = X
    
    if scaling==True:
        for i in range(len(X)):
            X[i] = X[i] / np.std(X)
    else:
        X = X
        
    Z = X
    return Z

def compute_covariance_matrix(Z):

    COV = np.matmul(np.matrix.transpose(Z), Z)
    return COV

def find_pcs(COV):
    
    PCS, L = np.linalg.eig(COV)
    
    sort = PCS.argsort()[::-1]
    PCS = PCS[sort]; L = L[:,sort]
    
    return PCS, L

def project_data(Z, PCS, L, k, var):
    
    if k == 0:
        cumSumArray = np.cumsum(L) / L.sum()
        
        for i in range(len(cumSumArray)):
            if cumSumArray[i] >= var:
                cumSumInd = np.where(cumSumArray >= var)[0][0]
                u = PCS[:, cumSumInd]
                Z_star = np.matmul(Z, u)

    else:
        u = PCS[:, k-1]
        Z_star = np.matmul(Z, u) 
        
    Z_star = np.vstack(Z_star)

    return Z_star