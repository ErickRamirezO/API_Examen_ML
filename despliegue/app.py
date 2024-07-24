import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Paths del modelo, preprocesador y columnas preentrenados
MODEL_PATH = './modelo_svm.pkl'
SCALER_PATH = './preprocesador.pkl'
COLUMNS_PATH = './columns.pkl'

# Cargar el modelo, el preprocesador y las columnas
def load_model_and_scaler():
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    with open(SCALER_PATH, 'rb') as file:
        scaler = pickle.load(file)
    with open(COLUMNS_PATH, 'rb') as file:
        columns = pickle.load(file)
    return model, scaler, columns

# Preprocesar los datos
def preprocess(input_data, scaler, columns):
    # Separar variables categóricas y numéricas
    categorical_features = input_data.select_dtypes(include=['object']).columns
    numerical_features = input_data.select_dtypes(exclude=['object']).columns
    
    # Escalar variables numéricas
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])
    
    # Codificar variables categóricas
    input_data = pd.get_dummies(input_data, columns=categorical_features, drop_first=True)
    
    # Alinear columnas
    input_data = input_data.reindex(columns=columns, fill_value=0)
    
    return input_data

# Realizar la predicción
def model_prediction(processed_data, model):
    return model.predict(processed_data)

def main():
    # Cargar el modelo, el preprocesador y las columnas
    model, scaler, columns = load_model_and_scaler()

    # Título
    st.markdown("""
    <h1 style="color:#181082;text-align:center;">Sistema de Predicción de Ingresos</h1>
    """, unsafe_allow_html=True)

    # Entradas del usuario
    edad = st.number_input('Edad', min_value=0)
    clase_de_trabajo = st.selectbox('Clase de Trabajo', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    fnlwgt = st.number_input('$ Peso final', min_value=0)
    educacion = st.selectbox('Educación', ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '6th', '5th', '10th', '2nd', '1st', 'Preschool'])
    num_educativo = st.number_input('Número Educativo', min_value=0)
    estado_civil = st.selectbox('Estado Civil', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    ocupacion = st.selectbox('Ocupación', ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    relacion = st.selectbox('Relación', ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    raza = st.selectbox('Raza', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    genero = st.selectbox('Género', ['Female', 'Male'])
    ganancia_de_capital = st.number_input('Ganancia de Capital', min_value=0)
    pérdida_de_capital = st.number_input('Pérdida de Capital', min_value=0)
    horas_por_semana = st.number_input('Horas por Semana', min_value=0)
    país_nativo = st.selectbox('País Nativo', ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Brazil', 'Dominican-Republic', 'Columbia', 'Peru', 'Thailand', 'Ecuador', 'Taiwan', 'Haiti', 'Hungary', 'Greece', 'Nicaragua', 'Scotland', 'Belgium', 'Armenia', 'Serbia', 'Russia', 'El-Salvador', 'Lithuania', 'Syria', 'Ukraine', 'Estonia', 'Croatia', 'Lebanon', 'Slovakia', 'Slovenia', 'Bosnia-Herzegovina', 'Albania', 'Montenegro', 'North Korea', 'South Korea'])
    
    # Crear DataFrame con las entradas del usuario
    input_data = pd.DataFrame([[edad, clase_de_trabajo, fnlwgt, educacion, num_educativo, estado_civil, ocupacion, relacion, raza, genero, ganancia_de_capital, pérdida_de_capital, horas_por_semana, país_nativo]],
                              columns=['edad', 'clase_de_trabajo', 'fnlwgt', 'educación', 'num_educativo', 'estado_civil', 'ocupación', 'relación', 'raza', 'género', 'ganancia_de_capital', 'pérdida_de_capital', 'horas_por_semana', 'país_nativo'])
    
    # Preprocesar datos
    processed_data = preprocess(input_data, scaler, columns)
    
    # Realizar predicción
    if st.button('Predecir'):
        prediction = model_prediction(processed_data, model)
        st.success('El ingreso previsto es: {}'.format('>50K' if prediction[0] == '>50K' else '<=50K'))

if __name__ == '__main__':
    main()
