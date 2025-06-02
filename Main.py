import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data
data= load_data('./Titanic-Dataset.csv')
#print(data.shape)
def preprocess_data(data):
    # Extract titles from names before dropping the Name column
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4,
        'Dr': 5, 'Rev': 5, 'Colonel': 5, 'Major': 5, 'Mlle': 2,
        'Countess': 3, 'Ms': 2, 'Lady': 3, 'Sir': 1, 'Mme': 3, 'Capt': 5
    }
    data['Title_encoded'] = data['Title'].map(title_mapping)
    
    # Drop unnecessary columns after extracting titles
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Handle categorical variables
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    
    # Fill missing values in numerical columns
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    
    # Fill missing values in Embarked with most frequent value
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    
    # Convert Embarked to dummy variables
    data = pd.get_dummies(data, columns=['Embarked'], prefix=['Embarked'])
    
    # Add new features
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    return data


def create_pipeline():
    numeric_features = ['Age', 'Fare', 'FamilySize', 'Title_encoded']
    categorical_features = ['Sex', 'Pclass']  # Removed 'Embarked'

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))  
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            learning_rate=0.01,
            n_estimators=100,
            max_depth=3,
            random_state=42,
            n_iter_no_change=10,
            validation_fraction=0.1,
            tol=1e-4
        ))
    ])

    return full_pipeline


# Load and prepare data
data = load_data('./Titanic-Dataset.csv')
data = preprocess_data(data)

# Split features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train pipeline
pipeline = create_pipeline()
pipeline.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")
