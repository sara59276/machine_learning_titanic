import pandas as pd
import statsmodels.api as sm
from constants.constants import PROJECT_ROOT



train_df = pd.read_csv(
    PROJECT_ROOT + '/resource/titanic/train.csv',
    converters={'Sex': lambda x: 1 if x == 'female' else 0},
)
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median()) # TODO justifying this filling method
train_df['Embarked_S'] = (train_df['Embarked'] == 'S').astype(int)
train_df['Embarked_C'] = (train_df['Embarked'] == 'C').astype(int)
train_df['Embarked_Q'] = (train_df['Embarked'] == 'Q').astype(int)

y = train_df["Survived"].astype(int)
X = train_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_S", "Embarked_C", "Embarked_Q"]]
X = sm.add_constant(X) # adding an intercept # TODO : why important ?

glm = sm.GLM(
    y,
    X,
    family=sm.families.Binomial(), # TODO justify binomial (VD is binary)
)
results = glm.fit()
print(results.summary())

# TODO séparer resource en 2-3 parties (train + test) + validate?

results.save(fname="logistic_regression_model")

probs = results.predict()


