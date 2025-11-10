from sklearn.decomposition import PCA
import numpy as np
def pca_train(data,n_components=2):
    pca = PCA(n_components=n_components)
    pca_model=pca.fit(data)
    pca_vector=pca_model.components_.T
    pca_variance=pca_model.explained_variance_
    pca_variance_ratio = pca_model.explained_variance_ratio_*100
    latent_data = np.matmul(data, pca_vector)

    return pca_model,pca_vector,pca_variance,pca_variance_ratio,latent_data



def pca_train_test(): #Avoid using main function
    # 假資料
    data = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12],])

    pca_components=pca_train(data)
if __name__=='__main__':
    pca_train_test()