import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

def normalize_min_max(matrix: np.ndarray) -> np.ndarray:
    """
        Método para normalizar los datos
        
        Args: 
            1. matrix (array): Matriz de datos original

        Returns: 
            1. normalized_data: Matriz de datos normalizados
    """
    max_values = np.max(matrix, axis=0)
    min_values = np.min(matrix, axis=0)

    # Lleva los datos al hiperplano (0, 1) utilizando la normalización min-max
    normalized_data = (matrix - min_values) / (max_values - min_values)
    return normalized_data

def calculate_covariance(ma: np.ndarray, mb: np.ndarray = None) -> np.ndarray:
    """
    Calcula la matriz de covarianza entre las filas de las matrices ma y mb.

    Parameters:
    - ma (np.ndarray): Matriz de datos.
    - mb (np.ndarray, optional): Otra matriz de datos. Si no se proporciona, se asume que es igual a ma.

    Returns:
    - np.ndarray: Matriz de covarianza entre las filas de ma y mb.
    """
    if mb is None:
        mb = ma
    filas_ma, columnas_ma = ma.shape
    filas_mb, columnas_mb = mb.shape

    if columnas_ma != columnas_mb:
        raise ValueError("Las matrices deben tener el mismo número de columnas")

    covarianzas = np.zeros((filas_ma, filas_mb))

    for i in range(filas_ma):
        for j in range(filas_mb):
            covarianza = np.sum((ma[i, :] - np.mean(ma[i, :])) * (mb[j, :] - np.mean(mb[j, :]))) / (columnas_ma - 1)
            covarianzas[i, j] = covarianza

    return covarianzas

def send_to_box(datapoint: float, ranges: np.ndarray) -> int:
    """
    Determina el índice de la caja a la que pertenece un datapoint en función de los rangos especificados.

    Parameters:
    - datapoint (float): Punto de datos a asignar a una caja.
    - ranges (np.ndarray): Rangos que definen las cajas.

    Returns:
    - int: Índice de la caja a la que pertenece el datapoint.
    """
    index = None
    for i in range(len(ranges) - 1):
        if datapoint >= ranges[i] and datapoint < ranges[i + 1]:
            index = i
    return index

def particiones_vertices(rn: int, a: float, b: float, tamano_deseado: int) -> np.ndarray:
    """
    Genera todas las posibles particiones de vértices en un espacio de rn dimensiones, con coordenadas
    que varían entre a y b en pasos de tamaño especificado.

    Parameters:
    - rn (int): Número de dimensiones.
    - a (float): Valor mínimo de las coordenadas.
    - b (float): Valor máximo de las coordenadas.
    - tamano_deseado (int): Tamaño deseado de las particiones.

    Returns:
    - np.ndarray: Matriz que representa todas las posibles particiones de vértices.
    """

    n_particiones = int(tamano_deseado ** (1 / rn))
    points = np.linspace(a, b, n_particiones)
    permutations = list(itertools.product(points, repeat=rn))

    return np.array(permutations)

def create_folder_if_not_exists(folder_path: str) -> None:
    """
    Crea un directorio si no existe.

    Args:
        folder_path (str): Ruta del directorio.

    Returns:
        None
    """
    os.makedirs(folder_path, exist_ok=True)

def plotting_clusters(clusters: dict, data: np.ndarray, folder_path: str, parameters: dict) -> None:
    """
    Genera un gráfico 3D para visualizar clusters en un conjunto de datos.

    Args:
        clusters (dict): Un diccionario donde las claves son los identificadores de los clusters y los valores son índices de datos asociados a cada cluster.
        data (array): Matriz de datos de forma (n_samples, n_features), donde las columnas representan características del conjunto de datos.
        folder_path (str): Path de la carpeta donde se van a guardar los resultados
        parameters (dict): Diccionario con la especificacion de los parametros

    Returns:
        None
    """
    filas, columnas = data.shape
    if columnas>3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for idx, val in clusters.items():
            value = data[val]
            ax.scatter(value[:, 1], value[:, 2], value[:, 3], label=idx)

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.legend()

        parameters_string = '_'.join([f"{key}={value}" for key, value in parameters.items()])
        plt.savefig(f'{folder_path}/{parameters_string}.png')
        plt.close()
    else: 
        parameters_string = '_'.join([f"{key}={value}" for key, value in parameters.items()])
        for idx, val in clusters.items():
            value = data[val]
            plt.plot(value[:,0],value[:,1],'o', label=idx)
        plt.savefig(f'{folder_path}/{parameters_string}.png')
        plt.close()

def convert_to_serializable(data):
    if isinstance(data, np.float32):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    return data