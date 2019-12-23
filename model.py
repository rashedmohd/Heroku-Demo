# importing the dataset 
import pandas as pd
import numpy 
from sklearn import preprocessing
import pickle 
  
df = pd.read_csv("G:\\Data science ExcelR\\Project\\Project_Claims.csv")     
df.head() 

df = df.drop(['Unnamed: 0', 'Region','State'], axis = 1)

df.describe()

## filling NA values
df.isnull().sum()

## filling NA values in Claim_Value with median
df["Claim_Value"].fillna(df.Claim_Value.median(),inplace=True)  ##median of claim value is 7370 

df.columns
col_names = df.columns 

category_col =['Area', 'City', 'Consumer_profile', 'Product_category', 'Product_type', 'Purchased_from', 'Purpose']  

labelEncoder = preprocessing.LabelEncoder() 

mapping_dict ={} 
for col in category_col: 
    df[col] = labelEncoder.fit_transform(df[col]) 
  
    le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
  
    mapping_dict[col]= le_name_mapping 
print(mapping_dict)

from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 

X = df.values[:, 0:17] 
Y = df.Fraud.values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 
  
dt_clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                     random_state = 100, 
                                     max_depth = 5, 
                                     min_samples_leaf = 5) 
  
dt_clf_gini.fit(X_train, y_train) 
y_pred_gini = dt_clf_gini.predict(X_test) 
  
print ("Desicion Tree using Gini Index\nAccuracy is ", 
             accuracy_score(y_test, y_pred_gini)*100 ) 

## Saving model to disk
pickle.dump(dt_clf_gini, open('model.pkl','wb'))

## loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(X_test))




