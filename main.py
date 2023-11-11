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

##4(a) Base DT
penguin_base_dt = DecisionTreeClassifier()
penguin_base_dt.fit(x_train_penguin,y_train_penguin)

abalone_base_dt = DecisionTreeClassifier()
abalone_base_dt.fit(x_train_abalone,y_train_abalone)

plt.figure(figsize=(20, 10))
plot_tree(penguin_base_dt, feature_names=x_train_penguin.columns, class_names=list(map(str, penguin_base_dt.classes_)), filled=True, rounded=True)
plt.savefig('penguin_base_dt.png', format='png')
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(abalone_base_dt, feature_names=x_train_abalone.columns, class_names=list(map(str, abalone_base_dt.classes_)), filled=True, rounded=True, max_depth=5)
plt.savefig('abalone_base_dt.png', format='png')
plt.show()

#5
y_penguin_pred = penguin_base_dt.predict(x_test_penguin)
y_abalone_pred = abalone_base_dt.predict(x_test_abalone)

print("-------- Base-DT Penguin Performance-------- \nConfusion Matrix:")
print(confusion_matrix(y_test_penguing, y_penguin_pred))
print("\nClassification Report:")
print(classification_report(y_test_penguing, y_penguin_pred))

print("-------- Base-DT Abalone Perfromance:--------  \nConfusion Matrix:")
print(confusion_matrix(y_test_abalone, y_abalone_pred))
print("\nClassification Report:")
print(classification_report(y_test_abalone, y_abalone_pred))








