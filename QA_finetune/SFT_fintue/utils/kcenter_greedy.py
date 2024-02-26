from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import pairwise_distances
from sample_def import SamplingMethod

class kCenterGreedy(SamplingMethod):

  def __init__(self, X, metric='euclidean'):
    self.X = X
    self.flat_X = self.flatten_X()
    self.name = 'kcenter'
    self.features = self.flat_X
    self.metric = metric
    self.min_distances = None
    self.n_obs = self.X.shape[0]
    self.already_selected = []
    print('shape of features')
    print(X.shape)

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.

    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:
      # Update min_distances for all examples given new cluster center.
      x = self.features[cluster_centers]
      dist = pairwise_distances(self.features, x, metric=self.metric)

      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)

  def select_batch_(self, features, already_selected, N, **kwargs):
    """
    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.

    Args:
      model: model with scikit-like API with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to minimize distance to cluster centers
    """
    #if already_selected is None:
        #already_selected = []
     #   already_selected = [ index for index in np.random.choice(np.arange(self.n_obs),200,replace=False)]
    try:
      # Assumes that the transform function takes in original data and not
      # flattened data.
      print('Getting transformed features...')
      self.features = features
      print('Calculating distances...')
      self.update_distances(already_selected, only_new=False, reset_dist=True)
    except:
      print('Using flat_X as features.')
      self.update_distances(already_selected, only_new=True, reset_dist=False)

    if already_selected is None:
        already_selected = []
     #   already_selected = np.random.choice(np.arange(self.n_obs),100,replace=False)
    self.already_selected = already_selected
    print(self.already_selected)

    new_batch = []

    for _ in range(N):
      if self.already_selected == []:
        # Initialize centers with a randomly selected datapoint
        ind = np.random.choice(np.arange(self.n_obs))
      else:
        ind = np.argmax(self.min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.
      assert ind not in already_selected
      
      if self.min_distances is None:
        print('min distances is None')
      else:
        print('Maximum distance from cluster centers is %0.2f'
            % max(self.min_distances))
      
      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
      
      if self.already_selected is None:
          self.already_selected = []
      else:
          self.already_selected.append(ind)

    print('Maximum distance from cluster centers is %0.2f'
            % max(self.min_distances))
    
    return self.already_selected