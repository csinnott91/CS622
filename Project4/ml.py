from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import f1_score

def dt_train(X, Y):
    dt = DecisionTreeClassifier()
    dt = dt.fit(X, Y)
    
    return dt

def kmeans_train(X):
    kmeans = KMeans(n_clusters = 2).fit(X)
    
    return kmeans

def knn_train(X, Y, K):
    knn = NearestNeighbors(X, n_neighbors = K, algorithm = 'kd_tree')
    knn.fit(X, Y)
    
    return knn

def perceptron_train(X, Y):
    perc = Perceptron()
    perc.fit(X, Y)
    
    return perc

def nn_train(X, Y, hls):
    nn = MLPClassifier(hidden_layer_sizes = hls)
    nn.fit(X, Y)
    
    return nn

def pca_train(X, K):
    pca = PCA(n_components = K)
    pca.fit(X)
    
    return pca

def pca_transform(X, pca):
    pca_transform = pca.transform(X)
    
    return pca_transform


def svm_train(X, Y, k):
    svm = SVC()
    svm.fit(X, Y)
    
    return svm

def model_test(X, model):
    test_predict = model.predict(X)
 
    return test_predict

def compute_F1(Y, Y_hat):
    f1 = f1_score(Y, Y_hat, average = "micro")
    
    return f1
