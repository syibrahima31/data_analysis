import numpy as np
from sklearn.datasets import load_iris


X = load_iris().data 


class Kmeans:
    def __init__(self, k: int, max_iter: int):
        """
        Args:
            k (int): represente le nombre de cluster 
            max_iter (int): le nombre maximum d'ittération 
        """
        self.k = k
        self.max_iter = max_iter
        
        # initialisation des clusters 
        self.clusters = [[] for _ in range(self.k)]
        
        # initalisations des centroids 
        self.centroids = None

    def fit(self, features: np.ndarray):
        """
        Args:
            features (np.ndarray): la matrice des variables explicative 
        """
        self.features = features

        self.nrow, self.ncol = features.shape

        # etape 1 : initialisation des centroides
        index_centroids = np.random.choice(self.nrow, self.k, replace=False)
        self.centroids = features[index_centroids]

     
        # la boucle d'execution de l'algo

        for _ in range(self.max_iter):
            # Etape 1 : mise a jour des clusters
            print(_)

            self.clusters = self.make_clusters(self.centroids)

            previous_centroid = self.centroids
            # Etape 2 : update des centroids

            self.centroids = self.update_centroids(self.clusters)

            # Etape 3 : regarder est ce que l'algo converge
            
            if self.converge(previous_centroid, self.centroids) == 0 :
                break 
    
    
    def converge(self, prev_centroid, new_centroid):
        return sum(self.dist_euclid(prev_centroid[i], new_centroid[i]) for i in range(self.k))
    
            
       
    
    
    
    def update_centroids(self, clusters): 
        
        centroids = np.zeros(shape=(self.k, self.ncol))
        
        for c in range(self.k): 
            index_c = clusters[c]
            center_c = self.features[index_c].mean(axis=0)
            centroids[c] = center_c 
        
        return centroids 
        
    
    
    
    

    def make_clusters(self, centroids:np.ndarray) -> list:
        """_summary_

        Args:
            centroids (np.ndarray): le tableau contenant les centres 

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





if __name__ == "__main__": 
    model = Kmeans(3, 100)
    model.fit(X)
    print(model.centroids)
