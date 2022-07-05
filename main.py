import numpy as np
from clustering import kmeans
from sklearn.datasets import load_iris


# chargement du jeu de données
features = load_iris().data


# cest le nombre de clusters
k = 3

# c'est le nombre maximal d'ittération
max_iter = 300

model = kmeans.Kmeans(k=k, max_iter=max_iter)
clusters = model.fit(features)


if __name__ == "__main__":
    print(clusters)
