import pandas as pd
from sklearn.model_selection import train_test_split

from constants.constants import PROJECT_ROOT

# PassengerId

### y, target, Variable dépendante ###
# Survived

### X, features, Variables indépendantes ###
# Pclass
# Sex : converted male to 0 and female to 1
# Age
# SibSp
# Parch
# Fare
# Embarked : One-Hot Encoding

### dropped variables ### # TODO : justifier
# Name
# Ticket
# Cabin

def process_titanic():
    # load resource
    titanic_df = pd.read_csv(
        PROJECT_ROOT + '/resource/titanic/train.csv',
        converters={'Sex': lambda x: 1 if x == 'female' else 0},
    )

    # handle missing resource
    titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].median())  # TODO change to linear regression

    # handle categorial features
    titanic_df['Embarked_S'] = (titanic_df['Embarked'] == 'S').astype(int)
    titanic_df['Embarked_C'] = (titanic_df['Embarked'] == 'C').astype(int)
    titanic_df['Embarked_Q'] = (titanic_df['Embarked'] == 'Q').astype(int)

    return titanic_df

def load_titanic():
    titanic_df = process_titanic()

    # split resource into features (VI, X) and target (VD, y)
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_S", "Embarked_C", "Embarked_Q"]
    X = titanic_df[feature_cols]
    y = titanic_df["Survived"].astype(int)

    return X, y

def load_titanic_split_sets(test_size: float):
    X, y = load_titanic()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    return X_train, X_test, y_train, y_test
