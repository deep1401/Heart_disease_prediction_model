import pandas as pd
data_path='https://s3.amazonaws.com/drivendata/data/54/public/train_values.csv'
train_data=pd.read_csv(data_path)
train_data=pd.DataFrame(train_data)
labels=pd.read_csv('https://s3.amazonaws.com/drivendata/data/54/public/train_labels.csv')
labels=pd.DataFrame(labels)
features=['age','sex','slope_of_peak_exercise_st_segment','chest_pain_type','num_major_vessels','fasting_blood_sugar_gt_120_mg_per_dl','oldpeak_eq_st_depression','max_heart_rate_achieved','exercise_induced_angina']
X=train_data[features]
y=labels.heart_disease_present
test_data=pd.read_csv('https://s3.amazonaws.com/drivendata/data/54/public/test_values.csv')
test_data=pd.DataFrame(test_data)
n_data=test_data[features]
from xgboost import XGBRegressor
my_model = XGBRegressor()
my_model.fit(X,y,verbose=False)
predictions = my_model.predict(n_data)
output=pd.DataFrame({'patient_id':labels.patient_id,'heart_disease_present':prediction})
filename = 'Submission_deep1401.csv'

submission.to_csv(filename,index=False)
