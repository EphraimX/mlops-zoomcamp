import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from prefect import get_run_logger

from datetime import datetime, date

import pickle


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):

    # df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    # df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)


    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def get_paths(date_value=None):

    if date_value == None:
        date_value = date.today().strftime("%Y-%m-%d")

    new_date = date_value.split('-')
    year, month, day = new_date[0], new_date[1], new_date[2]
    year, month, day = int(year), int(month), int(day)

    new_date = date(year, month, day)
        

    if new_date.month != 1 and new_date.month != 2:
        train_month , train_year = (new_date.month-2, new_date.year)
    elif new_date.month == 1:
        train_month , train_year = (11, new_date.year-1)
    else:
        train_month , train_year = (12, new_date.year-1)

    train_date = new_date.replace(day=1, month=train_month, year=train_year)
    train_date = train_date.strftime('%Y-%m')
        

    val_month , val_year = (new_date.month-1, new_date.year) if new_date.month != 1 else (12, new_date.year-1)
    val_date = new_date.replace(day=1, month=val_month, year=val_year)
    val_date = val_date.strftime('%Y-%m')

    train_path = f'./data/fhv_tripdata_{train_date}.parquet'
    val_path = f'./data/fhv_tripdata_{val_date}.parquet'

    return train_path, val_path


@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return 


@flow
def main(date_=None):

    
    train_path, val_path = get_paths(date_value=date_).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path).result()
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path).result()
    df_val_processed = prepare_features(df_val, categorical, False).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    # save the model

    with open(f"models/model-{date_}.bin", "wb") as lr_out:
        pickle.dump(lr, lr_out)
    
    with open(f"models/dv-{date_}.b", "wb") as dv_out:
        pickle.dump(dv, dv_out)

    run_model(df_val_processed, categorical, dv, lr)


# model deployment using CronSchedule

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner


DeploymentSpec(
    name="cron-model-deployment",
    flow=main,
    schedule = CronSchedule(
        cron = "0 9 15 * *",
        timezone = "Africa/Lagos"),
        flow_runner=SubprocessFlowRunner(),
        tags=["ml"]

)