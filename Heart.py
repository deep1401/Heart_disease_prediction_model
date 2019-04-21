import pandas as pd
data_path='https://s3.amazonaws.com/drivendata/data/54/public/train_values.csv'
train_data=pd.read_csv(data_path)
train_data=pd.DataFrame(train_data)
thel={'normal':1,'fixed_defect':2,'reversible_defect':3}        #categorical encoding of variable for better logloss
train_data.thal = [thel[item] for item in train_data.thal]       
labels=pd.read_csv('https://s3.amazonaws.com/drivendata/data/54/public/train_labels.csv')
labels=pd.DataFrame(labels)
features=['age','slope_of_peak_exercise_st_segment','thal','chest_pain_type','fasting_blood_sugar_gt_120_mg_per_dl','oldpeak_eq_st_depression','max_heart_rate_achieved','exercise_induced_angina']
X=train_data[features]
y=labels.heart_disease_present
test_data=pd.read_csv('https://s3.amazonaws.com/drivendata/data/54/public/test_values.csv')
test_data=pd.DataFrame(test_data)
thel={'normal':1,'fixed_defect':2,'reversible_defect':3}        
test_data.thal = [thel[item] for item in test_data.thal]   
n_data=test_data[features]
from xgboost import XGBRegressor
my_model = XGBRegressor()
my_model.fit(X,y,verbose=False)
prediction = my_model.predict(n_data)
output=pd.DataFrame({'patient_id':test_data.patient_id,'heart_disease_present':prediction})
filename = 'Submission7_deep1401.csv'
output.to_csv(filename,index=False)

#Current LogLoss 0.4449
