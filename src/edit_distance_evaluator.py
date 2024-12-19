'''
Como los LLMs a veces no se ajustan a lo que se les pide y devuelven más caracteres
como clase de salida, usamos como clase predicha aquella con menor distancia de edición
(en pocas palabra, la más parecida entre las clases posibles).
'''
from sklearn.metrics import classification_report

def edit_distance(s1, s2):    
    # Inicializamos la matriz de distancias
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    # Inicializamos la primera fila
    for i in range(len(s1) + 1):
        dp[i][0] = i

    # Inicializamos la primera columna
    for j in range(len(s2) + 1):
        dp[0][j] = j

    # Calculamos la matriz de distancias
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + (s1[i - 1] != s2[j - 1]),
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1
            )

    # Devolvemos la distancia de edición
    return dp[-1][-1]

def evaluate(y_true, y_pred):
    # Obtenemos todos los valores únicos de y_true
    unique_values = y_true.unique()

    # Para aquellos valores de y_pred que no se encuentren en unique_values,
    # buscamos el valor más cercano por distancia de edición
    y_pred = y_pred.apply(lambda x: min(unique_values, key=lambda y: edit_distance(x, y)))
    return classification_report(y_true, y_pred, output_dict=True)