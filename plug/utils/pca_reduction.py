import numpy as np
from sklearn.decomposition import PCA as sklearnPCA

def pca_reduction(Sm,n_comp):
    '''#### Perform PCA-based mechanism reduction #############################
    Input:
        Sm = sensitivity matrix for molar/mass fractions at various temperatures
        n_comp = desired number of principal components
        
    Output:
        eigenvalues = PCA eigenvalues
        eigenvectors = PCA eigenvectors
    ########################################################################'''

    #Set number of principal components
    sklearn_pca = sklearnPCA(n_components=n_comp) 
        
    #This function assumes that the inert species is always at the last column
    #Remove inert species from sensitivity matrix
    data = Sm
    
#    #Swap axes of original array and reshape it to a 2D matrix
#    data = np.swapaxes(data,0,1)
#    data = np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2])).T

    #Scattered matrix
    scatter_data = np.dot(data.T,data)
            
    #Compute the PCA
    sklearn_pca.fit(scatter_data)
    
    #Get the components of each principal axis
    eigenvectors = sklearn_pca.components_.T
    
    #Get associated eigenvalues
    eigenvalues = sklearn_pca.explained_variance_
                    
    #Output 
    out = (eigenvalues,eigenvectors)
    
    return out
