import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

##Reading Data------------
penguin_data = pd.read_csv('Content/penguins.csv')
abalone_data = pd.read_csv('Content/abalone.csv')

##Turingin Data to numerical------------
# onehot_encoder = OneHotEncoder(sparse=False)
# onehot_endoded = onehot_encoder.fit_transform(penguin_data[['island', 'sex']])
# onehot_penguin_data = pd.DataFrame(onehot_endoded, columns=onehot_encoder.get_feature_names_out(['island', 'sex']))
# penguin_data = pd.concat([penguin_data.drop(['island', 'sex'], axis=1), onehot_penguin_data], axis=1)
penguin_data['island'] = penguin_data['island'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
penguin_data['sex'] = penguin_data['sex'].map({'MALE': 0, 'FEMALE': 1})

##Graphs of data----------------------
penguin_species_distribution = penguin_data['species'].value_counts(normalize=True) * 100
plt.figure(figsize=(10, 6))
penguin_species_distribution.plot(kind='bar', color='skyblue')
plt.title('Penguin Species Distribution')
plt.xlabel('Species')
plt.ylabel('Percentage of instances')
plt.xticks(rotation=45, ha='right')  
plt.savefig('penguin-classes.png', format='png')
plt.show()

abalone_sex_distribution = abalone_data['Type'].value_counts(normalize=True) * 100
plt.figure(figsize=(10, 6))
abalone_sex_distribution.plot(kind='bar', color='skyblue')
plt.title('Abalone Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Percentage of instances')
plt.xticks(rotation=45, ha='right')  
plt.savefig('abalone-classes.png', format='png')
plt.show()

##Spliting data-------------
x_penguin = penguin_data.drop('species', axis=1)
y_penguin = penguin_data['species']
x_train_penguin, x_test_penguin, y_train_penguin, y_test_penguing = train_test_split(x_penguin, y_penguin, test_size=0.2, random_state=42)

x_abalone = abalone_data.drop('type', axis=1)
y_abalone= penguin_data['type']
x_train_abalone, x_test_abalone, y_train_abalone, y_test_abalone = train_test_split(x_abalone, y_abalone, test_size=0.2, random_state=42)

##Doing 4(a) and a little bit of 5-------------------------
penguin_base_dt = DecisionTreeClassifier()
penguin_base_dt.fit(x_train_penguin,y_train_penguin)

y_test_predict_proba = penguin_base_dt.predict_proba(X_test)

# calc confusion matrix
y_test_predict = tree.predict(X_test[columns])
print("Confusion Matrix Tree : \n", confusion_matrix(y_test, y_test_predict),"\n")
print("The precision for Tree is ",precision_score(y_test, y_test_predict)) 
print("The recall for Tree is ",recall_score(y_test, y_test_predict),"\n")  








