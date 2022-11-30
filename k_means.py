import numpy as np
import math
import copy


class KMeans():

    def __init__(self, input, k=2):
        self.input = np.array(input).T
        self.k = k

        self.centroids = [[0 for i in range(k)] for j in range(k)]

        for i in range(len(self.centroids)):
            for j in range(len(self.centroids)):
                self.centroids[i][j] = self.input[i][j]

    def fit(self, input, epoch=10):
        input = np.array(input).T
        cluster = [[], []]
        self.clusters = [[], []]
        while epoch:
            cluster[0].clear()
            cluster[1].clear()
            for i in input:
                d1 = self._ecludian_distance(
                    self.centroids[0][0], self.centroids[0][1], i[0], i[1])
                d2 = self._ecludian_distance(
                    self.centroids[1][0], self.centroids[1][1], i[0], i[1])
                if d1 > d2:
                    cluster[0].append([i[0], i[1]])
                else:
                    cluster[1].append([i[0], i[1]])

            x, y = 0, 0
            for i in cluster[0]:
                x += i[0]
                y += i[1]
            self.centroids[0] = [x/len(cluster[0]), y/len(cluster[0])]

            x, y = 0, 0
            for i in cluster[1]:
                x += i[0]
                y += i[1]
            self.centroids[1] = [x/len(cluster[1]), y/len(cluster[1])]
            if (self.clusters == cluster):
                break
            else:
                self.clusters = cluster
            epoch -= 1
        return cluster

    def _ecludian_distance(self, cx, cy, x, y):
        return math.sqrt(((cx-x)**2 + (cy-y)**2))


input = [[1.0, 1.5, 3.0, 5.0, 3.5, 4.5, 3.5, 3.6, 4.2, 6.5, 7.2, 5.6, 4.3, 5.8, 4.6],
         [1.0, 2.0, 4.0, 7.0, 5.0, 5.0, 4.5, 5.2,
             8.2, 7.8, 3.5, 8.5, 4.6, 9.5, 5.5]
         ]
km = KMeans(input, k=2)
cluster = km.fit(input)
print(cluster)
