import numpy as np

# Turn on scikit-learn optimizations with these 2 simple lines:

from sklearnex import patch_sklearn

patch_sklearn()


# Import scikit-learn algorithms after the patch is enabled 

from sklearn.cluster import KMeans


X = np.array([[1, 2], [1, 4], [1, 0],

 [10, 2], [10, 4], [10, 0]])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print(f"kmeans.labels_ = {kmeans.labels_}")
