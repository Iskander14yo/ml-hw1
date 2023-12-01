import pickle
import numpy as np
import pandas as pd


# Определяем функцию для удаления единиц измерения и преобразования в float
def process_column(column: pd.Series) -> pd.Series:
    return column.str.extract(r'(\d+\.\d+|\d+)')[0].astype(float)


# Определяем функцию подготовки данных
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df['mileage'] = process_column(df['mileage'])
    df['engine'] = process_column(df['engine'])
    df['max_power'] = process_column(df['max_power'])
    df.drop('torque', axis=1, inplace=True)

    # Вычисление медиан и заполнение ими пропусков
    with open('pkl/medians.pkl', 'rb') as f:
        medians = pickle.load(f)
    df[['mileage', 'engine', 'max_power', 'seats']] = df[['mileage', 'engine',
                                                          'max_power', 'seats']].fillna(medians)
    df['engine'] = df['engine'].astype(int)

    # Отранжировали year на начальный год
    df.insert(0, "age", df["year"].max() + 1 - df["year"])
    df.drop('year', axis=1, inplace=True)

    # добавили бренд машины, ее название удалили - слишком частые категории
    df['brand'] = df.name.apply(lambda x: x.split()[0])
    df.drop('name', axis=1, inplace=True)

    df['seats'] = df['seats'].astype('object')

    with open('pkl/ohe_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    df_cat = df.select_dtypes(['object'])
    df_cat_encoded = encoder.transform(df_cat)
    encoded_feature_names = encoder.get_feature_names_out(df_cat.columns)
    df_cat_encoded_df = pd.DataFrame(df_cat_encoded, columns=encoded_feature_names)
    df = pd.concat([df_cat_encoded_df, df.select_dtypes(exclude='object')], axis=1)

    return df


def make_inference(df: pd.DataFrame) -> np.array:
    df = prepare_data(df)
    with open('pkl/model.pkl', 'rb') as f:
        model = pickle.load(f)
    pred = model.predict(df)
    return pred
