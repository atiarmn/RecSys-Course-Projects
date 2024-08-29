from lenskit.algorithms import basic, bias, user_knn, item_knn, als, svd
from recipetag import recipeTagMF
algorithms = {
    'POP': basic.Popular(),
    'BIAS': bias.Bias(),
    'U-KNN': user_knn.UserUser(30),
    'I-KNN': item_knn.ItemItem(20, 2, save_nbrs=5000),
    'I-KNNs': item_knn.ItemItem(20, 2, feedback='implicit', save_nbrs=5000),
    'E-MF': als.BiasedMF(50),
    'SVD': svd.BiasedSVD(50),
    'TAG': recipeTagMF(50, components="all", loss='bpr')
}

pred_algos = ['BIAS', 'U-KNN', 'I-KNN', 'E-MF', 'SVD']
