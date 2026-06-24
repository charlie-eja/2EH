from sklearn.decomposition import PCA
import numpy as np
from typing import Tuple
import pandas as pd
def pca_train(data : np.ndarray,
              n_components : int =2,
              index :str =None,
              save_or_not :int=0) -> Tuple[None,np.ndarray, np.ndarray,np.ndarray, np.ndarray]:
    if save_or_not==1:
        if data.shape[1]!=len(index):
            print('index and data must have same length')
        else:
            pc_list = [f"PC{i}" for i in range(1, data.shape[1] + 1)]
            pca = PCA(n_components=data.shape[1])
            pca_model = pca.fit(data)
            pca_vector = pca_model.components_.T
            df = pd.DataFrame(
                data=pca_vector,
                index=index,
                columns=pc_list
            )
            df.to_excel("pca vactor\pca_vactor.xlsx")
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
