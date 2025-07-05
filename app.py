import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Cargar el modelo y los encoders
nombre_archivo_modelo = 'modelo_naive_bayes.jb'
model = joblib.load(nombre_archivo_modelo)

# Definir y cargar los LabelEncoders
# Es crucial que estos LabelEncoders sean los mismos que se usaron para entrenar el modelo
# Aquí los recreamos manualmente basados en el orden en que fueron ajustados en el código original
label_encoders = {}
# Horas de Estudio: Alta -> 1, Baja -> 0
le_horas_estudio = LabelEncoder()
le_horas_estudio.fit(["Baja","Alta"]) # Asegurarse de que el orden sea el mismo
label_encoders["Horas de Estudio"] = le_horas_estudio

# Asistencia: Buena -> 1, Mala -> 0
le_asistencia = LabelEncoder()
le_asistencia.fit(["Mala","Buena"]) # Asegurarse de que el orden sea el mismo
label_encoders["Asistencia"] = le_asistencia

# Resultado: Sí -> 1, No -> 0
le_resultado = LabelEncoder()
le_resultado.fit(["No","Sí"]) # Asegurarse de que el orden sea el mismo
label_encoders["Resultado"] = le_resultado


# Título y subtítulo de la aplicación
st.title("Predicción de Clase")
st.markdown('<h2 style="color: red;">Elaborado por: Klaudialiliana</h2>', unsafe_allow_html=True)

st.write("Seleccione los valores de las variables para la predicción:")

# Entradas del usuario
horas_estudio_input = st.selectbox("Horas de Estudio", ["Alta", "Baja"])
asistencia_input = st.selectbox("Asistencia", ["Buena", "Mala"])

# Crear un DataFrame con las entradas del usuario
nueva_observacion = pd.DataFrame({
    "Horas de Estudio": [horas_estudio_input],
    "Asistencia": [asistencia_input]
})

# Codificar la nueva observación usando los LabelEncoders cargados/creados
nueva_observacion_codificada = nueva_observacion.copy()
for column in nueva_observacion_codificada.columns:
    nueva_observacion_codificada[column] = label_encoders[column].transform(nueva_observacion_codificada[column])

# Realizar la predicción al hacer clic en un botón
if st.button("Predecir"):
    # Realizar la predicción
    prediccion_numerica = model.predict(nueva_observacion_codificada)

    # Decodificar la predicción numérica de vuelta a la etiqueta original
    prediccion_etiqueta = label_encoders["Resultado"].inverse_transform(prediccion_numerica)

    # Mostrar el resultado con emojis
    if prediccion_etiqueta[0] == "Sí":
        st.write(prediccion_numerica,nueva_observacion_codificada)
        st.success(f"Felicitaciones Aprueba! 😊")
    else:
        st.error(f"No aprueba 😞")
