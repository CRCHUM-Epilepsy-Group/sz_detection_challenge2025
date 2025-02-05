import numpy as np
from szdetect import model as mod


# Pull_features

# Long-to-wide pivot


# X_test = wide_df.drop(index_col)
# X_test = scaler.transform(X_test_rec) # TODO add scaling on training data
# y_pred = pretrained_model.predict(X_test)

# Set your search grid
tau_range = [3, 4, 5] 
thresh_range = [0.45, 0.55, 0.65, 0.75]

# Loop over all combinations of (tau, threshold)
# reg_pred = mod.firing_power(pred_rec, tau=tau)
# pred_ann = np.array([1 if x >= threshold else 0 for x in pred_ann])


# calculate epoch-based metrics