
import pickle
import numpy as np
import pywt
import matplotlib.pyplot as plt

with open('WISDM.pkl', 'rb') as f:
    X, y = pickle.load(f)

X = np.array(X)
y = np.array(y)
print(X.shape)

# def normalize_list(lst):
#     min_val = min(lst)
#     max_val = max(lst)
#     normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
#     return normalized_lst


# def normalize(matrix):
#     min_val = np.min(matrix)
#     max_val = np.max(matrix)
#     normalized_matrix = (matrix - min_val) / (max_val - min_val)
#     return normalized_matrix







# def get_cwt(index):

#     # set up wavelet parameters
#     wavelet = 'morl'
#     scales = np.arange(1, 51)

#     # perform CWT and get power spectrum for each component
#     power = []
#     for i in range(3):
#         coef, freqs = pywt.cwt(X[index][i], scales, wavelet)
#         power.append(abs(coef)**2)

#     return np.array(power)
# acc_x = X[0][0]
# acc_y = X[0][1]
# acc_z = X[0][2]
# cwt_x = get_cwt(0)[0]
# cwt_y = get_cwt(0)[1]
# cwt_z = get_cwt(0)[2]


# acc_x = normalize_list(acc_x)
# acc_y = normalize_list(acc_y)
# acc_z = normalize_list(acc_z)
# cwt_x = normalize(cwt_x)
# cwt_y = normalize(cwt_y)
# cwt_z = normalize(cwt_z)

# # class jogging

# # class jogging
# def plot(l, n, order):
#     plt.figure()  # create a new figure
#     x = range(1, 151)  # assuming 150 time steps
#     # plotting the time series
#     plt.plot(x, l)
#     plt.xlabel('Index of the Samples')
#     plt.ylabel(f'{n}')
#     plt.title(f'({order}) Jogging')
#     plt.savefig(f'figs/{n}.pdf')

# plot(acc_x, 'acceleration x-axis', 'a')
# plot(acc_y, 'acceleration y-axis', 'b')
# plot(acc_z, 'acceleration z-axis', 'c')




# # Plot the CWT matrix
# def plot_cwt_matrix(mat, n, order):
#     plt.figure()
#     plt.imshow(mat, cmap='jet', aspect='auto')
#     plt.colorbar()
#     plt.xlabel('Index of the Samples')
#     plt.ylabel('Wavelet Function Scale')
#     plt.title(f'({order}) Jogging')
#     plt.savefig(f'figs/{n}.pdf')


# plot_cwt_matrix(cwt_x, 'cwt acceleration x-axis', 'd')
# plot_cwt_matrix(cwt_y, 'cwt acceleration y-axis', 'e')
# plot_cwt_matrix(cwt_z, 'cwt acceleration z-axis', 'f')


