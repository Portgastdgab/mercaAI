import pandas as pd

# 1. Leer el archivo CSV original, asegurando que las columnas se lean correctamente
data = pd.read_csv('amazon_cells_labelled.csv', header=None)

# 2. Limpiar el texto de las columnas y eliminar filas donde cualquiera de las dos primeras columnas esté vacía
data[0] = data[0].astype(str).str.strip()  # Limpiar espacios en blanco
data[1] = data[1].astype(str).str.strip()  # Limpiar espacios en blanco

# 3. Filtrar filas para asegurarse de que ambas columnas (comentario y etiqueta) no estén vacías
data = data[(data[0] != '') & (data[1] != '')]

# 4. Asegurarse de que las etiquetas son solo 0 o 1
data = data[data[1].isin(['0', '1'])]

# 5. Seleccionar solo las dos primeras columnas (comentarios y etiquetas)
cleaned_data = data[[0, 1]]

# 6. Renombrar las columnas para mayor claridad
cleaned_data.columns = ['comment', 'label']

# 7. Guardar el nuevo DataFrame en un nuevo archivo CSV
cleaned_data.to_csv('amazon_cells_labelled_cleaned.csv', index=False)

print("Archivo CSV limpio creado: 'amazon_cells_labelled_cleaned.csv'")
