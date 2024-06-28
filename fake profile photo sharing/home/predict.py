import sys
import joblib
import pandas as pd


model = joblib.load('threshold1.pkl')


group_size = float(sys.argv[1])
accept_count = float(sys.argv[2])
reject_count = float(sys.argv[3])

feature_names = ['group_size', 'accept_count', 'reject_count']


input_data = pd.DataFrame([[group_size, accept_count, reject_count]], columns=feature_names)


prediction = model.predict(input_data)


print(prediction[0])
