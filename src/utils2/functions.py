# Import Libraries

from catboost import CatBoostRegressor, Pool
import datetime
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Functions

class Data_Operations():
    def __init__(self) -> None:
        pass
    
    def load_data(dataset):
        '''
        Esta función carga el dataset original para iniciar el preprocesado

        Input: Ruta de acceso (paths[0]) + archivo csv (data_files[0])
        Output: DataFrame Pandas (df)
        '''
        df = pd.read_csv(dataset)
        return df

    def clean_nans(df):
        '''
        Esta función elimina todos los NaN, ya que al estar la mayoria en la variable 'Rating' (Nuestro Target para
        la predicción), carece de sentido estimar sus valores durante el preprocesado y contamos con un volumen de 
        datos suficiente.
        
        Input: Dataframe original cargado desde el csv (df).
        Output: Otro Dataframe sin valores NaN.
        '''
        df_No_NaNs = df.copy().dropna(subset=["Rating","Content Rating", "Current Ver", "Android Ver"])
        df_No_NaNs.to_csv(r'data\processed_files\df_No_NaNs.csv', index = False)
        return df_No_NaNs
    
    def dataset_preprocessing(df_No_NaNs):
        '''
        Esta función limpia el dataset inicial para poder trabajar con el modelo (CatBoost).
        Transforma y cambia el tipo de:
        Crea una nueva variable a partir de:
        Elimina las variables:

        Input: dataset
        Output: Dataset limpio y listo para trabajar con el modelo (CatBoost)
        '''
        df_No_NaNs["Reviews"] = df_No_NaNs["Reviews"].astype('int64')
        df_No_NaNs["Size"] = df_No_NaNs["Size"].replace(['Varies with device'],['14000']).apply(lambda x: float(x.replace('M','')) *1000 if 'M' in x else (float(x.replace('k','')) /1000 if 'k' in x else x)).astype('float64')
        df_No_NaNs["Installs"] = df_No_NaNs["Installs"].str.split('+',expand=True)[0].apply(lambda x: x.replace(',','')).astype('int64')
        df_No_NaNs["Main_Genre"] = df_No_NaNs["Genres"].str.split(';',expand=True)[0]
        df_No_NaNs["Last Updated"] = df_No_NaNs["Last Updated"].apply(lambda x: x.replace(' ','/').replace(',','').replace('January','1').replace('February','2').replace('March','3').replace('April','4').replace('May','5').replace('June','6').replace('July','7').replace('August','8').replace('September','9').replace('October','10').replace('November','11').replace('December','12')).astype('datetime64')
        timestamp = pd.Timestamp(datetime.datetime(2021, 10, 10))
        df_No_NaNs['Today'] = pd.Timestamp(timestamp.today().strftime('%d-%m-%Y'))
        df_No_NaNs['Days_Since_Last_Update'] = (df_No_NaNs['Today'] - df_No_NaNs["Last Updated"]).dt.days
        df_No_NaNs.drop(['App','Type','Price','Genres','Today','Last Updated'], axis=1, inplace=True)
        df_Preprocessed = df_No_NaNs.copy()
        df_Preprocessed.to_csv(r'data\processed_files\df_Preprocessed.csv', index = False)
        return df_Preprocessed

class Model_Operations():
    def __init__(self) -> None:
        pass

    def split_test(df):
        X_train, X_test, y_train, y_test = train_test_split(df.drop('Rating', axis=1),
                                                        df['Rating'],
                                                        test_size=0.3,
                                                        random_state=42)
        return X_train, X_test, y_train, y_test
    
    def pre_fit(X_train, X_test, y_train, y_test):
        pool_train = Pool(X_train, y_train,
                            cat_features=['Category','Content Rating','Current Ver','Android Ver','Main_Genre'])
        pool_test = Pool(X_test, y_test,
                        cat_features=['Category','Content Rating','Current Ver','Android Ver','Main_Genre'])
        return pool_train, pool_test
             
    def train_model(train):
        cb = CatBoostRegressor(n_estimators=1000,
                      loss_function='RMSE',
                      learning_rate=0.1,
                      random_state=1,
                      verbose=False
                      )
        cb.fit(train)
        model_trained = cb
        return model_trained

    def rmse_score(model_trained, X,y):
        predict = model_trained.predict(X)
        rmse = mean_squared_error(y, predict)
        return rmse
    
    def save_best_model(model_trained):
        timestamp = pd.Timestamp(datetime.date(2021, 10, 10))
        fecha_hoy = timestamp.today().strftime(('%d-%m-%Y'))
        joblib.dump(model_trained, r'model\best_app_rating_model-'+fecha_hoy+'.pkl')

    def open_last_saved_model(pickle):
        final_model_reloaded = joblib.load(pickle)
        return final_model_reloaded
    
    def prepare_full_dataset(df_Preprocessed):
        X_full = df_Preprocessed.drop('Rating', axis=1)
        y_full = df_Preprocessed['Rating']
        return X_full, y_full

    def pre_fit_full_dataset(X_full, y_full):
        pool_full_dataset = Pool(X_full, y_full,
                            cat_features=['Category','Content Rating','Current Ver','Android Ver','Main_Genre'])
        return pool_full_dataset

    def save_best_model_full_dataset(model_trained_full_dataset):
        timestamp = pd.Timestamp(datetime.date(2021, 10, 10))
        fecha_hoy = timestamp.today().strftime(('%d-%m-%Y'))
        joblib.dump(model_trained_full_dataset, r'model\best_app_rating_model_full_training-'+fecha_hoy+'.pkl')