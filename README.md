# MIACE_py
Python Implementation of MI-ACE and MI-SMF Target Characterization Algorithms

NOTE: If MI-ACE or MI-SMF Algorithms are used in any publication or presentation, the following reference must be cited:
A. Zare, C. Jiao, T. Glenn, "Multiple Instance Hyperspectral Target Characterization," IEEE Trans. on Pattern Analysis and Machine Intelligence, To Appear.

The function to run either MI-SMF or MI-ACE:

opt_target, opt_obj_val, b_mu, sig_inv_half, init_t = mi_target(data_bags, labels, parameters)

Inputs:
- data_bags - cell array where each cell is a bag containing a NxD matrix of instances where N is the number of instances in the bag and D is the dimensionality of each instance
- labels - binary values, the same length as dataBags, indicates positive bag with '1' and negative bags with '0'
- parameters - parameters dictionary containing the following:
	- method_flag: Set to True for MI-ACE, Set to False for MI-SMF
	- global_background_flag: Set to True to use global mean and covariance, set to False to use negative bag mean and covariance
	- init_type: Type 1 is to use best positive instance based on objective function value
    			 Type 2 clusters the data with k-means and selects the best cluster center as the initial target signature
	- pos_label: Value used to indicate positive bags, usually 1
	- neg_label: Value used to indicate negative bags, usually 0 or -1
	- max_iter: Maximum number of iterations (rarely used)
	- sample_por: Percentage of positive data points used to initialize (default = 1)
	- init_k: If using init2, number of clusters used to initialize (default = 1000)

Outputs:
- opt_target: Estimated target concept
- opt_obj_val: Final Objective Function value
- b_mu: Background Mean to be used in ACE or SMF detector with test data
- sig_inv_half: Square root of background covariance, Use sig_inv_half.T @ sig_inv_half as covariance in ACE or SMF detector with test data
- init_t: initial target concept

Files explanation:
Latest Revision: Oct. 2018

demo_simple_example.py: Demo script to run MI-SMF and MI-ACE on simulated Hyperspectral Data
- detectors.py: SMF and ACE Target Detector Code
- LICENSE: License for this code
- mi_target.py: Main MI-SMF and MI-ACE function and implementation
- README.md: This file
- smf_det.m: SMF Target Detector Code
