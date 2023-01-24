#! /usr/bin/python

from .opt_forest import OptForest
from .sampling import VSSampling
from .opt import E2LSH
from .opt import AngleLSH
from scipy.spatial import distance

class OptIForest(OptForest):
	   
	"""
    OptIForest: Anomaly Detection Algorithm.
    Return the anomaly score of each sample. The lower, the more abnormal.

    Parameters
    ----------
	lsh_family : string, optional (default="L2SH"). The base detectors of OptIForest.
		The possible values include 'ALSH' for angular distance, 'L1SH' for L1 (Mahattan) distance, 'L2SH' for L2 (Euclidean) distance. The default value is 'L2SH' given this distance metric is commonly-used in real applications.
    num_trees : int, optional (default=100)
        The number of base estimators in the ensemble.
    granularity : int, optional (default=1)
        This parameter is to control the sensitivity of the algorithm with respect to duplicated or very similar data instances which can lead to only-one-partition case for LSH and are hard to be partitioned. If the value is '1', the model takes the lenghth of single branches of an isolation. Otherwise, the isolation will be 'virtually' compressed by just counting binary/multi-fork branches. 
    threshold: int, optional (default=403)
    branch: int, optional (default=0)

    Examples
    --------
    >>> from detectors import OptIForest
    >>> X = [[-1.1], [0.3], [0.5], [100]]
    >>> clf = OptIForest.fit(X)
    >>> clf.decision_function([[0.1], [0], [90]])
	>>> array([-0.21098136, -0.23885756, -0.71920724])
    """
	
	def __init__(self, lsh_family='L2OPT', num_trees=100, threshold=403, branch=0, granularity=1):
		if lsh_family == 'ALOPT':
			OptForest.__init__(self, num_trees, VSSampling(num_trees), AngleLSH(), threshold, branch, distance.cosine, granularity)
		elif lsh_family == 'L1OPT':
			OptForest.__init__(self, num_trees, VSSampling(num_trees), E2LSH(norm=1), threshold, branch, distance.cityblock, granularity)
		else:
			OptForest.__init__(self, num_trees, VSSampling(num_trees), E2LSH(norm=2), threshold, branch, distance.euclidean, granularity)