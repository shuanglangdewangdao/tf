import os
import pandas as pd #读取csv文件
import numpy as np
import pytz # 时区处理
import tensorflow as  tf
from sklearn import preprocessing # 数据标准化
from sklearn.model_selection import train_test_split
def process_data(file_path):
    file_name = os.path.split(file_path)[1]
    time_name = 'COLLECTTIME'
    data = pd.read_csv(file_path,parse_dates=[time_name])
    end_time = data.iloc[-1][3]
    data2 = data.set_index([time_name])
    data2 = data2.interpolate(method='time')
    data2['year'] = data2.index.year
    data2['month'] = data2.index.month
    data2['day'] = data2.index.day
    data2['weekday'] = data2.index.weekday
    data2['work'] = data2.index.weekday < 5
    data2['week'] = data2.index.week
    data2['work'] = data2['work'].astype(int)
    data2 = data2.reset_index()
def rescale(s):
    reshaped_s = s.values.reshape(-1, 1)
    scaler = preprocessing.StandardScaler().fit(reshaped_s)
    return pd.DataFrame(data=scaler.transform(reshaped_s)), scaler
def input_fn(df, label):
    #"""Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    
    
    continuous_cols = {
        k: tf.constant(df[k].values, 
            shape=[df[k].size, 1]) 
        for k in CONTINUOUS_COLUMNS
        }
    feature_cols = dict(continuous_cols)
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.

    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols.update(categorical_cols)
    # feature_cols=dict(categorical_cols)
    # Converts the label column into a constant Tensor.
#     label = tf.constant(df[LABEL_COLUMN].values, shape=[df[LABEL_COLUMN].size, 1])
    label = tf.constant(label.values, shape=[label.size, 1])
    # Returns the feature columns and the label.
    return feature_cols, label
def scale(df):
    for column in CONTINUOUS_COLUMNS:
        df[column], column = rescale(df[column])

    for column in CATEGORICAL_COLUMNS:
        if column:
            df[column] = df[column].apply(str)
    df[LABEL_COLUMN], label_scaler = rescale(df[LABEL_COLUMN])
    df['label'] = df[LABEL_COLUMN]
    return df, label_scaler
file_path = 'D:\\py\\tf\\DNNRegressor\\discdata_processed.csv'
df = process_data(file_path)
df2, label_scaler = scale(df.copy())
X_train, X_test, y_train, y_test = train_test_split(df2[FEATURE_COLUMNS], df2[LABEL_COLUMN], test_size=0.3)
def input_fn(df, label):
    #"""Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in CONTINUOUS_COLUMNS}
    feature_cols = dict(continuous_cols)
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.

    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols.update(categorical_cols)
    # feature_cols=dict(categorical_cols)
    # Converts the label column into a constant Tensor.
#     label = tf.constant(df[LABEL_COLUMN].values, shape=[df[LABEL_COLUMN].size, 1])
    label = tf.constant(label.values, shape=[label.size, 1])
    # Returns the feature columns and the label.
    return feature_cols, label


CATEGORICAL_COLUMNS = ['work', 'min'] #分类型字段
CONTINUOUS_COLUMNS = ['year', 'month', 'day', 'hour', 'weekday', 'week']#连续性字段
FEATURE_COLUMNS = []
FEATURE_COLUMNS.extend(CATEGORICAL_COLUMNS)
FEATURE_COLUMNS.extend(CONTINUOUS_COLUMNS)
LABEL_COLUMN = 'value'
deep_columns = []
for column in CATEGORICAL_COLUMNS:
    column = tf.contrib.layers.sparse_column_with_hash_bucket(
        column, hash_bucket_size=1000)
    deep_columns.append(tf.contrib.layers.embedding_column(column, dimension=7),)
for column in CONTINUOUS_COLUMNS:
    column = tf.contrib.layers.real_valued_column(column)
    deep_columns.append(column)
    model_dir = "./model" #模型保存位置
    learning_rate = 0.001 #学习速率

    #激活函数
    model_optimizer = tf.train.AdamOptimizer(
              learning_rate=learning_rate) 
   #定义模型            
m = tf.contrib.learn.DNNRegressor(model_dir=model_dir,
                                          feature_columns=deep_columns,
                                            hidden_units=[32, 64],
                                            optimizer=model_optimizer)
m.fit(input_fn=lambda: input_fn(X_train, y_train),
              steps=2500
                    )
ev=m.evaluate(input_fn=lambda: input_fn(X_test, y_test), steps=1)
print('ev: {}'.format(ev))
