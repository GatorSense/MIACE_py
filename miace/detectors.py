import numpy as np

def smf_det(hsi_data, tgt_sig, mu=None, sig_inv=None, tgt_flag=False):
"""
SMF Spectral Matched Filter

Inputs:
  hsi_data - DxN array of N data points of dimensionality D
  tgt_sig - Dx1 vector containing the target signature
  mu - Dx1 vector containing the background mean vector, if empty,
    computed as mean of all hsi_data
  sig_inv - DxD matrix containing the inverse background covariance, if
    empty, computed from all hsi_data
  tgt_flag - flag indicating whether mean should be subtracted from
  	target signatures or not, set to anything if mean should be subtracted.
Outputs:
  smf_data - Nx1 vector of SMF confidence values corresponding to each test point
  mu - Dx1 vector containing the background mean vector
  sig_inv - DxD matrix containing the inverse background covariance
"""
  # check for empty parameters
  mu = mu if mu else np.mean(hsi_data, axis=2)
  sig_inv = sig_inv if sig_inv else np.linalg.pinv(np.cov(hsi_data.T))

  # check target flag
  s = tgt_sig - mu if tgt_flag else tgt_sig

  # subtract the average from the dataset
  z = hsi_data - mu

  st_sig_inv = s.T * sig_inv
  st_sig_inv_s = st_sig_inv * s

  A = np.sum(st_sig_inv*z, axis=1)
  B = np.sqrt(st_sig_inv_s)

  smf_data = A / B
  
  return smf_data, mu, sig_inv

def ace_det(hsi_data, tgt_sig, mu=None, sig_inv=None, tgt_flag=False):
"""
ACE Adaptive Cosine Estimator

Inputs:
  hsi_data - DxN array of N data points of dimensionality D
  tgt_sig - Dx1 vector containing the target signature
  mu - Dx1 vector containing the background mean vector, if empty,
      computed as mean of all hsi_data
  sig_inv - DxD matrix containing the inverse background covariance, if
      empty, computed from all hsi_data
  target_flag - flag indicating whether mean should be subtracted from
  	target signatures or not, set to anything if mean should be subtracted.
Outputs:
  ace_data - Nx1 vector of ACE confidence values corresponding to each
      test point
  mu - Dx1 vector containing the background mean vector
  sig_inv - DxD matrix containing the inverse background covariance
"""
  # check for empty parameters
  mu = mu if mu else np.mean(hsi_data, axis=2)
  sig_inv = sig_inv if sig_inv else np.linalg.pinv(np.cov(hsi_data.T))

  # check target flag
  s = tgt_sig - mu if tgt_flag else tgt_sig

  # subtract the average from the dataset
  z = hsi_data - mu

  st_sig_inv = s.T * sig_inv
  st_sig_inv_s = st_sig_inv * s

  A = np.sum(st_sig_inv*z, axis=1)
  B = np.sqrt(st_sig_inv_s)
  C = np.sqrt(np.sum(z*siginv*z,axis=1));
  ace_data = A / (B * C)
  
  return ace_data, mu, sig_inv