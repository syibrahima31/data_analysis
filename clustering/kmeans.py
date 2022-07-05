
import numpy as np


class Kmeans:
    def __init__(self, k: int, max_iter: int):
        """
        Args:
            k (int): represente le nombre de cluster 
            max_iter (int): le nombre maximum d'ittération 
        """
        self.k = k
        self.max_iter = max_iter
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = None

    def fit(self, features: np.ndarray):
        """
        Args:
            features (np.ndarray): la matrice des variables explicative 
        """
        self.features = features

        self.nrow, self.col = features.shape

        # etape 1 : initialisation des centroides
        index_centroids = np.random.choice(self.nrow, self.k, replace=False)
        self.centroids = features[index_centroids]

        # la boucle d'execution de l'algo

        for _ in range(self.max_iter):
            # Etape 1 : mise a jour des clusters

            self.clusters = self.make_clusters(self.centroids)

            # Etape 2 : update des centroids

            self.centroids = update_centroids(parameters)

            # Etape 3 : regarder est ce que l'algo converge
        return self.clusters

    def make_clusters(self, centroids) -> list:
        """_summary_

        Args:
            centroids (_type_): le tableau contenant les centres 

        Returns:
            list : la liste des clusters 
        """

        clusters = [[] for _ in range(self.k)]

        for index, indiv in enumerate(self.features):
            num_cluster = self.search_index(indiv, centroids)
            clusters[num_cluster].append(index)
        return clusters

    def search_index(self, indiv, centroids):
        distance = [self.dist_euclid(indiv, center) for center in centroids]
        index = np.argmin(distance)
        return index

    def dist_euclid(self, x: np.ndarray, y: np.ndarray) -> float:
        """retoyrne la distance entre les vecteurs x et y 
        Args:
            x (np.ndarray): cest un vecteur 
            y (np.ndarray): c'est un vecteur 
        Returns:
            float: la distance 
        """
        return np.sqrt(np.sum((x-y)**2))
