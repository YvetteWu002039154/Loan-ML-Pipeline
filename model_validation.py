import joblib
import pandas as pd
import numpy as np

pipe = joblib.load('models/best_model.joblib')

incoming = np.array([22.0, "male", "Master", 66135.0, 1, "RENT",
                     35000.0, "MEDICAL", 14.27, 0.53, 4.0, 586, "NO"], dtype=object)

# prepare a DataFrame with the same columns/order as `features`
arr = np.asarray(incoming)
if arr.ndim == 1:
    arr = arr.reshape(1, -1)

pred = pipe.predict(arr)
print('predicted label indices:', pred)