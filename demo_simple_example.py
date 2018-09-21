import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from miace.detectors import ace_det, smf_det
from miace.mi_target import mi_target

results = {'smf': {}, 'ace': {}}

parameters = {
    "globalBackgroundFlag": False,
    "softmaxFlag": False,
    "posLabel": 1,
    "negLabel": 0,
    "maxIter": 100,
    "initType": 1
}


example_data = np.load('datasets/simple_example_data.npz')

# to accomodate bags of difference sizes
bag_list = [example_data['data_bags'][i] for i in range(example_data['data_bags'].shape[0])]

#to test whether the methods work on different sizes. combining a few bags together to create uneven bags
bag_list[0] = np.vstack((bag_list[0], bag_list[1]))
bag_list.pop(1)
labels = np.delete(example_data['labels'], 1, 1)

# SMF init1
parameters['methodFlag'] = False
parameters['samplePor'] = 1

# smf_opt_target, _, b_mu, sig_inv_half, _ = mi_target(
#     example_data['data_bags'], example_data['labels'], parameters)

smf_opt_target, _, b_mu, sig_inv_half, _ = mi_target(
    np.asarray(bag_list), labels, parameters)

print('Detecting SMF...')
smf_data, _, _ = smf_det(example_data['x_test'], smf_opt_target.T,
                         b_mu, np.matmul(sig_inv_half.T, sig_inv_half))

print('------------Method 2-------------')
# ACE init 1
parameters['methodFlag'] = True
parameters['samplePor'] = 1

# ace_opt_target, _, b_mu, sig_inv_half, _ = mi_target(
#     example_data['data_bags'], example_data['labels'], parameters)

ace_opt_target, _, b_mu, sig_inv_half, _ = mi_target(
    np.asarray(bag_list), labels, parameters)

print('Detecting ACE...')
ace_data, _, _ = ace_det(example_data['x_test'], ace_opt_target.T,
                         b_mu, np.matmul(sig_inv_half.T, sig_inv_half))

# Get false and true positive rate
labels_point_test = example_data['labels_point_test']

smf_fpr, smf_tpr, smf_threshold = roc_curve(
    labels_point_test, smf_data, pos_label=1)
ace_fpr, ace_tpr, ace_threshold = roc_curve(
    labels_point_test, ace_data, pos_label=1)


f, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=False)

# Target Concept
ax1.plot(smf_opt_target, 'b', label='SMF Target Concept')
ax1.plot(ace_opt_target, 'r', label='ACE Target Concept')
ax1.legend(loc='lower right')
ax1.set_xlabel('Band Number')

# ROC
ax2.plot(smf_fpr, smf_tpr, 'b', label='SMF ROC')
ax2.plot(ace_fpr, ace_tpr, 'r', label='ACE ROC')
ax2.plot([0, 1], [0, 1], 'r--')
ax2.legend(loc='lower right')
ax2.set_xlabel('Probability of False Alarm')
ax2.set_ylabel('Probability of Detection')

ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.show()
