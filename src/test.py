import numpy as np
import matplotlib.pyplot as plt
npz_file = np.load('outputs/ds19/feature/raw_feature_0-100x0-100.npz')

# Access the saved arrays using their respective keys
features = npz_file['features']
labels = npz_file['labels']

print(features.shape)

zero_counts = [np.count_nonzero(row == 0) for row in features]

# Plotting the histogram
plt.hist(zero_counts, bins='auto')  # 'auto' lets matplotlib decide the number of bins
plt.title('Histogram of Zero Padding in burst sequences')
plt.xlabel('Number of Zeros Added')
plt.ylabel('Frequency')
plt.savefig('outputs/ds19/stats/zero_traces.png')
plt.show()
