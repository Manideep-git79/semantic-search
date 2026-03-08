import numpy as np
import skfuzzy as fuzz

def perform_clustering():

    embeddings = np.load("embeddings/news_embeddings.npy")

    embeddings = embeddings.T

    clusters = 10

    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        embeddings,
        clusters,
        m=2,
        error=0.005,
        maxiter=1000
    )

    np.save("clustering/membership.npy",u)

    print("Fuzzy clustering done")

if __name__ == "__main__":
    perform_clustering()