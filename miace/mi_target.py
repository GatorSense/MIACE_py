import numpy as np

# default parameters for miTarget
default_parameters = {
    "methodFlag": True, # Set to 0 for MI-SMF, Set to 1 for MI-ACE
    "globalBackgroundFlag": False, # Set to 1 to use global mean and covariance, set to 0 to use negative bag mean and covariance
    "initType": 0, # Options: 1, 2, or 3.  InitType 1 is to use best positive instance based on objective function value, type 2 is to select positive instance with smallest cosine similarity with negative instance mean, type 3 clusters the data with k-means and selects the best cluster center as the initial target signature
    "softmaxFlag": False, # Set to 0 to use max, set to 1 to use softmax in computation of objective function values of positive bags
    "posLabel": 1, # Value used to indicate positive bags, usually 1
    "negLabel": 0, # Value used to indicate negative bags, usually 0 or -1
    "maxIter": 1000, # Maximum number of iterations (rarely used)
    "samplePor": 1, # Percentage of positive data points used to initialize (default = 1)
    "initK": 1000, # If using init3, number of clusters used to initialize (default = 1000)
    "numB": 5 # Number of background clusters (and optimal targets) to be estimated
}

def mi_target(data_bags, labels, parameters=default_parameters):
  """
  MIACE/MISMF Multiple Instance Adaptive Cosine Estimator/Multiple Instance
      Spectral Matched Filter Demo

  Inputs:
    data_bags - 1xB cell array where each cell contains an NxD matrix of N data points of
        dimensionality D (i.e.  N pixels with D spectral bands, each pixel is
        a row vector).  Total of B cells. 
    labels - 1XB array containing the bag level labels corresponding to
        each cell of the dataBags cell array
    parameters - struct - The struct contains the following fields:
      1. parameters.methodFlag: Set to 0 for MI-SMF, Set to 1 for MI-ACE
      2. parameters.initType: Options are 1, 2, or 3. 
      3. parameters.globalBackgroundFlag: set to 1 to use global mean and covariance, set to 0 to use negative bag mean and covariance
      4. parameters.softmaxFlag: Set to 0 to use max, set to
          1 to use softmax in computation of objective function
          values of positive bags  (This is generally not used
          and fixed at 0)
      5. parameters.posLabel: Value used to indicate positive
      bags, usually 1
      6. parameters.negLabel: Value used to indicate negative bags, usually 0 or -1
      7. parameters.maxIter: Maximum number of iterations (rarely used)
      8. parameters.samplePor: If using init1, percentage of positive data points used to initialize (default = 1) 
      9. parameters.initK = 1000; % If using init3, number of clusters used to initialize (default = 1000);
  Outputs:
    endmembers - double Mat - NxM matrix of M endmembers with N spectral
        bands
    P - double Mat - NxM matrix of abundances corresponding to M input
        pixels and N endmembers
  """
  num_bags = data_bags.shape[0]
  num_dim = data_bags.shape[2]
  num_pos_bags = np.sum(labels == parameters["posLabel"])
  data = data_bags if parameters["globalBackgroundFlag"] else data_bags[labels == parameters["negLabel"]]
  data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
  b_mu = np.mean(data, axis=0)
  b_cov = np.cov(data.T)

  """
  TODO: 
  whitening
  initializing
  optimizer
  undo whitening
  """
  # Whitening
  u, s, v = np.linalg.svd(b_cov)
  sig_inv_half = (1.0 / np.sqrt(s)) * np.identity(len(s)) * u.T
  m_scale = (data_bags - b_mu)*sig_inv_half.T
  denom = np.array([np.sqrt(np.sum(m_scale[i]*m_scale[i], axis=1)) for i in range(m_scale.shape[0])])

def init1():
  pass

def init2():
  pass

def init3():
  pass
  
def evalObjectiveWhitened():
  pass

if __name__ == "__main__":
  mi_target(np.random.random([5,5,5]), np.array([1,0,1,0,1]))
  


    