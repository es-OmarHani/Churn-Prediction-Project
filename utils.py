## main
import pandas as pd
import os


## skelarn -- preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector

## Read the Dataset
TRAIN_PATH = os.path.join(os.getcwd(), 'Churn_Modelling.csv')
df = pd.read_csv(TRAIN_PATH)

## Drop first 3 features
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

## Filtering using Age Feature using threshold
df.drop(index=df[df['Age'] > 80].index.tolist(), axis=0, inplace=True)


## To features and target
X = df.drop(columns=['Exited'], axis=1)
y = df['Exited']

## Split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=45, stratify=y)

## Slice the lists
num_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
categ_cols = ['Gender', 'Geography']

ready_cols = list(set(X_train.columns.tolist()) - set(num_cols) - set(categ_cols))


## Pipeline

## Numerical: num_cols --> Imputing using median, and standardscaler
## Categorical: categ_cols ---> Imputing using mode, and OHE
## Ready_cols ---> Imputing mode

## For Numerical
num_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(num_cols)),
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ])


## For Categorical
categ_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(categ_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('ohe', OneHotEncoder(drop='first', sparse_output=False))
                    ])


## For ready cols
ready_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(ready_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent'))
                    ])



## combine all
all_pipeline = FeatureUnion(transformer_list=[
                                    ('numerical', num_pipeline),
                                    ('categorical', categ_pipeline),
                                    ('ready', ready_pipeline)
                                ])

## apply, I need fitting only
all_pipeline.fit_transform(X_train)



def process_new(X_new):

    ## To get the columns to be able to call the pipeline (DataFrameSelector: requires columns names)
    df_new = pd.DataFrame([X_new])
    df_new.columns = X_train.columns

    ## Adjust the datatypes
    df_new['CreditScore'] = df_new['CreditScore'].astype('float')
    df_new['Geography'] = df_new['Geography'].astype('str')
    df_new['Gender'] = df_new['Gender'].astype('str')
    df_new['Age'] = df_new['Age'].astype('float')
    df_new['Tenure'] = df_new['Tenure'].astype('float')
    df_new['Balance'] = df_new['Balance'].astype('float')
    df_new['NumOfProducts'] = df_new['NumOfProducts'].astype('float')
    df_new['HasCrCard'] = df_new['HasCrCard'].astype('float')
    df_new['IsActiveMember'] = df_new['IsActiveMember'].astype('float')
    df_new['EstimatedSalary'] = df_new['EstimatedSalary'].astype('float')

    ## If you make Feature Engineering  -- here is its place


    ## Apply the pipeline
    X_processed = all_pipeline.transform(df_new)


    return X_processed