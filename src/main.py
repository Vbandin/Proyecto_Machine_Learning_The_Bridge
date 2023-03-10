from utils2.functions import Data_Operations, Model_Operations

# Cargo el Dataset Original

df = Data_Operations.load_data(r'data\raw_files\googleplaystore.csv')
print('Dataset Cargado')
# Elimino los NaN (La mayoria estan en el target y tenemos datos suficientes).

df_No_NaNs = Data_Operations.clean_nans(df)
print('Dataset sin NaNs')

# Completo el preprocesado de los datos 

df_Preprocessed = Data_Operations.dataset_preprocessing(df_No_NaNs)
print('Preprocesado acabado')

# Divido el dataset preprocesado en train y test

X_train, X_test, y_train, y_test = Model_Operations.split_test(df_Preprocessed)
print('División en Train y Test')

# Preparo datos para el entrenamiento

pool_train, pool_test = Model_Operations.pre_fit(X_train, X_test, y_train, y_test)
print('Datos listos para entrenamiento')

# Entreno el modelo

model_trained = Model_Operations.train_model(pool_train)
print('Modelo entrenado')

# Mido su eficacia en train (RMSE)

rmse_train_score = Model_Operations.rmse_score(model_trained, X_train,y_train)
print('RMSE Score del conjunto de train:', rmse_train_score)

# Mido su eficacia en test (RMSE)

rmse_test_score = Model_Operations.rmse_score(model_trained, X_test,y_test)
print('RMSE Score del conjunto de test:', rmse_test_score)

# Guardo el modelo

Model_Operations.save_best_model(model_trained)
print('Modelo guardado')

# Cargo el modelo guardado

final_model_reloaded = Model_Operations.open_last_saved_model(r"model\best_app_rating_model-10-03-2023.pkl")
print('Modelo recargado')

# Preparo el dataset completo para probar el modelo

X_full,y_full = Model_Operations.prepare_full_dataset(df_Preprocessed)
print('Conjunto de datos completo listo para probar el modelo')

# Pruebo el modelo guardado con todo el dataset y mido su eficacia con el dataset completo

rmse_full_score = Model_Operations.rmse_score(final_model_reloaded, X_full,y_full)
print('RMSE Score del dataset completo:', rmse_full_score)

# Reentreno el modelo con el dataset completo

pool_full_dataset = Model_Operations.pre_fit_full_dataset(X_full, y_full)
print('Conjunto de datos completo listo para reentrenar modelo')

model_trained_full_dataset = Model_Operations.train_model(pool_full_dataset)
print('Modelo final entrenado')

Model_Operations.save_best_model_full_dataset(model_trained_full_dataset)
print('Modelo final guardado')