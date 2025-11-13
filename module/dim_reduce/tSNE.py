from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from module.Utilities import  normalize

def  tsne_train(data,n_components=2,perplexity=30,learning_rate='auto',max_iter=1000,
        init='pca',metric='euclidean',random_state=42,verbose=1,):
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        init=init,
        metric=metric,
        random_state=random_state,
        verbose=verbose, #Display progress information ,set 1 ---> 50 steps error /per time
    )
    data_tsne = tsne.fit_transform(data)
    return data_tsne


def tsne_test():
    x, y = load_iris(return_X_y=True)
    normalize_data, mean_data, std_data =normalize.normalize_gaussian(x)
    data_tsne = tsne_train(normalize_data)
if __name__=='__main__':
    tsne_test()

