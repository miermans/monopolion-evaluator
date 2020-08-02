# The model code in  this file is derived from the TensorFlow feature_columns example:
# @see https://github.com/tensorflow/docs/blob/master/site/en/tutorials/structured_data/feature_columns.ipynb
#
# Copyright 2019 The TensorFlow Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column


class Classifier:
    NUMERIC_COLUMN_SUFFIXES = ['cash', 'score']
    TARGET_COLUMN = 'winningPlayer'

    METRICS = [
        tf.keras.metrics.Recall(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    ]

    def __init__(self, train_df: pd.DataFrame, validation_df: pd.DataFrame, player_count: int):
        self.train_df = train_df
        self.__add_engineered_features(self.train_df)
        self.validation_df = validation_df
        if self.validation_df is not None:
            self.__add_engineered_features(self.validation_df)

        self.headers = train_df.columns.values
        self.player_count = player_count
        self.__category_vocabulary_lists = {
            'die1': range(1, 7),
            'die2': range(1, 7),
            'position': range(40),
            'isInJail': range(2),
            'remainingTurnsInJail': range(4),
            'isOwned': range(2),
            'owner': range(-1, player_count),
            'buildingCount': range(5),
            'setCount': range(11),
        }

    def fit_model(self, epochs: int = 10, layers=None, batch_size: int = 100, learning_rate=0.001, dropout=0.2):
        if layers is None:
            layers = [512, 128]

        train_ds = self.df_to_dataset(self.train_df, batch_size=batch_size)
        validation_ds = None
        if self.validation_df is not None:
            validation_ds = self.df_to_dataset(self.validation_df)

        feature_layer = tf.keras.layers.DenseFeatures(self.get_feature_columns())

        model = tf.keras.Sequential()
        model.add(feature_layer)
        for units in layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            if dropout > 0:
                model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=Classifier.METRICS)

        model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=epochs)

        return model

    def get_feature_columns(self):
        cols = []
        for header in self.headers:
            if header == self.TARGET_COLUMN:
                continue
            elif any(header.endswith(s) for s in self.NUMERIC_COLUMN_SUFFIXES):
                cols.append(feature_column.numeric_column(header))
            else:
                vocab = self.__get_category_vocabulary_list(header)
                categorical_column = feature_column.categorical_column_with_vocabulary_list(header, vocab)
                cols.append(feature_column.indicator_column(categorical_column))

        return cols

    def df_to_dataset(self, data_frame: pd.DataFrame, shuffle: bool = True, batch_size: int = 100) -> tf.data.Dataset:
        """
        A utility method to create a tf.data dataset from a Pandas Dataframe
        :param data_frame:
        :param shuffle:
        :param batch_size:
        :return:
        """
        df = data_frame.copy()
        labels = df.pop(self.TARGET_COLUMN)
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df))
        ds = ds.batch(batch_size)
        return ds

    def __get_category_vocabulary_list(self, header):
        try:
            suffix = header.split('_')[-1]
            return self.__category_vocabulary_lists[suffix]
        except KeyError:
            return self.train_df[header].unique()

    def __add_engineered_features(self, df):
        pass
        # df['set1_owner'] = np.where(df['property_0_owner'] == df['property_1_owner'], df['property_0_owner'], -1)
        # df['set2_owner'] = np.where((df['property_3_owner'] == df['property_4_owner']) & (df['property_4_owner'] == df['property_5_owner']), df['property_3_owner'], -1)
        # df['set3_owner'] = np.where((df['property_6_owner'] == df['property_8_owner']) & (df['property_8_owner'] == df['property_9_owner']), df['property_6_owner'], -1)
        # df['set4_owner'] = np.where((df['property_11_owner'] == df['property_12_owner']) & (df['property_12_owner'] == df['property_13_owner']), df['property_11_owner'], -1)
        # df['set5_owner'] = np.where((df['property_14_owner'] == df['property_15_owner']) & (df['property_15_owner'] == df['property_16_owner']), df['property_14_owner'], -1)
        # df['set6_owner'] = np.where((df['property_18_owner'] == df['property_19_owner']) & (df['property_19_owner'] == df['property_21_owner']), df['property_18_owner'], -1)
        # df['set7_owner'] = np.where((df['property_22_owner'] == df['property_23_owner']) & (df['property_23_owner'] == df['property_24_owner']), df['property_22_owner'], -1)
        # df['set8_owner'] = np.where(df['property_26_owner'] == df['property_27_owner'], df['property_27_owner'], -1)
        # df['set9_owner'] = np.where((df['property_2_owner'] == df['property_10_owner']) & (df['property_10_owner'] == df['property_17_owner']) & (df['property_17_owner'] == df['property_25_owner']), df['property_2_owner'], -1)
        # df['set10_owner'] = np.where(df['property_7_owner'] == df['property_20_owner'], df['property_7_owner'], -1)
        # for p in range(2):
        #     df[f'player_{p}_setCount'] = 0
        #     for i in range(1, 11):
        #         df[f'player_{p}_setCount'] += (df[f'set{i}_owner'] == p).astype(int)
        #
        #     df[f'player_{p}_score'] = (2000 * df[f'player_{p}_setCount']) + df[f'player_{p}_cash']

        #df[f'game_owner'] = np.where(df[f'player_0_score'] >= df[f'player_1_score'], 0, 1)
