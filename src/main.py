import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils import resample
from functions import *
from sklearn.model_selection import train_test_split

num_iterations = 5

penguins = pd.read_csv('datasets/penguins.csv')
abalone = pd.read_csv('datasets/abalone.csv')

## Turning Data to numerical------------
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_endoded = onehot_encoder.fit_transform(penguins[['island', 'sex']])
onehot_penguins = pd.DataFrame(onehot_endoded, columns=onehot_encoder.get_feature_names_out(['island', 'sex']))
penguins = pd.concat([penguins.drop(['island', 'sex'], axis=1), onehot_penguins], axis=1)

# penguins['island'] = penguins['island'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
# penguins['sex'] = penguins['sex'].map({'MALE': 0, 'FEMALE': 1})

X_penguin = penguins.drop('species', axis=1) # Features
y_penguin = penguins['species'] # Target
X_train_penguin, X_test_penguin, y_train_penguin, y_test_penguin = train_test_split(X_penguin, y_penguin, test_size=0.3, random_state=42)

X_abalone = abalone.drop('Type', axis=1) # Features
y_abalone = abalone['Type'] # Target
X_train_abalone, X_test_abalone, y_train_abalone, y_test_abalone = train_test_split(X_abalone, y_abalone, test_size=0.2, random_state=42)

plot(penguins, 'penguin', 'species')
plot(abalone, 'abalone', 'Type')



# Base-DT
for iteration in range(num_iterations):
    accuracy_penguin, macro_f1_penguin, weighted_f1_penguin = base_dt(
        'performance/penguin-performance.txt', 
        'images/base_dt_penguin.png', 
        X_train_penguin, 
        X_test_penguin, 
        y_train_penguin, 
        y_test_penguin, 
        iteration
    )
    accuracy_abalone, macro_f1_abalone, weighted_f1_abalone = base_dt(
        'performance/abalone-performance.txt', 
        'images/base_dt_abalone.png', 
        X_train_abalone, 
        X_test_abalone, 
        y_train_abalone, 
        y_test_abalone, 
        iteration
    )

save_variance_to_file(
    'performance/penguin-performance.txt', 
    'Base-DT', 
    accuracy_penguin, 
    macro_f1_penguin, 
    weighted_f1_penguin
)
save_variance_to_file(
    'performance/abalone-performance.txt', 
    'Base-DT', 
    accuracy_abalone, 
    macro_f1_abalone, 
    weighted_f1_abalone
)

# Top-DT
for iteration in range(num_iterations):
    accuracy_penguin, macro_f1_penguin, weighted_f1_penguin = top_dt(
        'performance/penguin-performance.txt', 
        'images/top_dt_penguin.png', 
        X_train_penguin, 
        X_test_penguin, 
        y_train_penguin, 
        y_test_penguin, 
        iteration
    )
    accuracy_abalone, macro_f1_abalone, weighted_f1_abalone = base_dt(
        'performance/abalone-performance.txt', 
        X_train_abalone, 
        X_test_abalone, 
        y_train_abalone, 
        y_test_abalone, 
        iteration
    )

save_variance_to_file(
    'performance/penguin-performance.txt', 
    'Top-DT', 
    accuracy_penguin, 
    macro_f1_penguin, 
    weighted_f1_penguin
)
save_variance_to_file(
    'performance/abalone-performance.txt', 
    'Top-DT', 
    accuracy_abalone, 
    macro_f1_abalone, 
    weighted_f1_abalone
)

# Base-MLP
for iteration in range(num_iterations):
    accuracy_penguin, macro_f1_penguin, weighted_f1_penguin = base_mlp(
        'performance/penguin-performance.txt', 
        X_train_penguin, 
        X_test_penguin, 
        y_train_penguin, 
        y_test_penguin, 
        iteration
    )
    
    accuracy_abalone, macro_f1_abalone, weighted_f1_abalone = base_mlp(
        'performance/abalone-performance.txt', 
        X_train_abalone, 
        X_test_abalone, 
        y_train_abalone, 
        y_test_abalone, 
        iteration
    )

save_variance_to_file(
    'performance/penguin-performance.txt', 
    'Base-MLP', 
    accuracy_penguin, 
    macro_f1_penguin, 
    weighted_f1_penguin
)
save_variance_to_file(
    'performance/abalone-performance.txt', 
    'Base-MLP', 
    accuracy_abalone, 
    macro_f1_abalone, 
    weighted_f1_abalone
)

# Top-MLP
for iteration in range(num_iterations):
    accuracy_penguin, macro_f1_penguin, weighted_f1_penguin = top_mlp(
        'performance/penguin-performance.txt', 
        X_train_penguin, 
        X_test_penguin, 
        y_train_penguin, 
        y_test_penguin, 
        iteration
    )
    accuracy_abalone, macro_f1_abalone, weighted_f1_abalone = top_mlp(
        'performance/abalone-performance.txt', 
        X_train_abalone, 
        X_test_abalone, 
        y_train_abalone, 
        y_test_abalone, 
        iteration
    )

save_variance_to_file(
    'performance/penguin-performance.txt', 
    'Top-MLP', 
    accuracy_penguin, 
    macro_f1_penguin, 
    weighted_f1_penguin
)
save_variance_to_file(
    'performance/abalone-performance.txt', 
    'Top-MLP', 
    accuracy_abalone, 
    macro_f1_abalone, 
    weighted_f1_abalone
)
