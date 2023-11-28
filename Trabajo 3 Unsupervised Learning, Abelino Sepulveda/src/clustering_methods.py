import copy
import numpy as np
from typing import Dict, List
from src.distance_methods import metricas
from src.auxiliar_methods import send_to_box, particiones_vertices

def separate_boxes_by_datapoint(matriz_distancia: np.ndarray, num_boxes: int, punto = None) -> dict:
    """
    Separa las cajas según un punto de datos en una matriz de distancias.

    Args:
    - matriz_distancia (np.ndarray): Matriz de distancias.
    - num_boxes (int): Número de cajas a crear.
    - punto (int): Índice del punto de datos para el cual se realizará la separación de cajas.

    Returns:
    - dict: Diccionario que representa las cajas y los puntos asociados a cada una.
    """
    if punto is None:
        punto = np.random.choice(np.arange(matriz_distancia.shape[0]),size=1)[0]
    min_value = np.min(matriz_distancia)
    max_value = np.max(matriz_distancia)
    ranges = np.linspace(min_value, max_value, num_boxes + 1)
    ranges[-1] += 0.2

    boxes = {f'cluster {i}': [] for i in range(len(ranges) - 1)}

    items = list(boxes.items())
    for index, datapoint in enumerate(matriz_distancia[punto]):
        indice = send_to_box(datapoint, ranges)
        _, valor = items[indice]
        valor.append(index)

    return boxes

def separate_boxes(matriz_distancia: np.ndarray, num_boxes: int) -> dict:
    """
    Separa las cajas según las distancias en una matriz y clasifica los puntos en cada caja.

    Args:
    - matriz_distancia (np.ndarray): Matriz de distancias.
    - num_boxes (int): Número de cajas a crear.
    - data (np.ndarray): Matriz de datos asociados a las distancias.

    Returns:
    - dict: Diccionario que representa las cajas y los puntos clasificados en cada una.
    """
    num_boxes +=1
    min_value = np.min(matriz_distancia)
    max_value = np.max(matriz_distancia)
    ranges = np.linspace(min_value, max_value, num_boxes + 1)
    ranges[-1] += 0.2

    boxes = {f'cluster {i}': [] for i in range(len(ranges) - 1)}
    classified_boxes = {f'cluster {i}': [] for i in range(len(ranges) - 1)}

    filas, columnas = matriz_distancia.shape
    items = list(boxes.items())
    
    for fila in range(filas):
        for columna in range(columnas):
            indice = send_to_box(matriz_distancia[fila, columna], ranges)
            _, valor = items[indice]
            valor.append({'distancia': matriz_distancia[fila, columna], 'indices': columna})

    indices = np.arange(filas)
    for indice_punto in indices:
        cantidad_apariciones = {}
        for key, value in boxes.items():
            cantidad_apariciones[key] = sum(1 for item in value if item['indices'] == indice_punto)
        clave_con_mayor_valor = max(cantidad_apariciones, key=cantidad_apariciones.get)
        classified_boxes[clave_con_mayor_valor].append(indice_punto)

    del classified_boxes[f"cluster {num_boxes-1}"]

    return classified_boxes

class Mountain:
    def __init__(self, alpha: float, metrica: str, args={} ) -> None:
        """
        Se inicializa la clase Mountain con el parámetro alpha.

        Args:
        - alpha (float): Parámetro para la función de montaña.
        """
        self.alpha = alpha
        self.beta = 1.5 * alpha
        self.args = args
        self.distance_method = metricas(metrica)

    def mountain_function(self, vertices: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Calcula la función de montaña para los vértices dados y los datos proporcionados.

        Args:
        - vertices (np.ndarray): Vértices para los cuales se calculará la función de montaña.
        - data (np.ndarray): Datos sobre los cuales se calculará la función de montaña.

        Returns:
        - np.ndarray: Peso asociado a cada vértice.
        """
        peso = np.zeros(len(vertices))
        for id_v, vi in enumerate(vertices):
            cont = 0
            for _, xi in enumerate(data):
                valor_distancia = self.distance_method(**{**{'ma':np.array([vi]),'mb':np.array([xi])},**self.args})[0,0]
                cont += np.exp(-((valor_distancia ** 2) / (2 * self.alpha ** 2)))
            peso[id_v] = cont
        return peso

    def seleccion_centros(self, data: np.ndarray, vertices: np.ndarray) -> Dict[str, Dict[str, int]]:
        """
        Selecciona los centros de los clusters utilizando la técnica de funciones de montaña.

        Args:
        - data (np.ndarray): Datos sobre los cuales se realizará el clustering.
        - vertices (np.ndarray): Vértices para los cuales se calcularán los centros.

        Returns:
        - Dict[str, Dict[str, int]]: Diccionario que contiene la información sobre los centros seleccionados.
        """
        mv = self.mountain_function(vertices, data)
        c_escogidos = {}
        idx_c1 = np.argmax(mv)
        c_escogidos['1'] = {'indice': idx_c1, 'm': mv[idx_c1]}

        bool = True
        cont = 2

        while bool:
            mc = vertices[idx_c1]
            indices = [valor['indice'] for valor in c_escogidos.values()]
            mv_center = copy.deepcopy(mv[idx_c1])
            for idx_m, mi in enumerate(vertices):
                valor_distancia = self.distance_method(**{**{'ma':np.array([mi]),'mb':np.array([mc])},**self.args})[0,0]
                mv[idx_m] = mv[idx_m] - mv_center * np.exp(-((valor_distancia ** 2) / (2 * self.beta ** 2)))
            idx_c1 = np.argmax(mv)
            if idx_c1 != indices[-1]:
                c_escogidos[str(cont)] = {'indice': idx_c1, 'm': mv[idx_c1]}
                cont += 1
            else:
                bool = False
        return c_escogidos

    def clustering(self, data: np.ndarray) -> Dict[str, List[int]]:
        """
        Realiza el clustering utilizando la técnica de funciones de montaña.

        Args:
        - data (np.ndarray): Datos sobre los cuales se realizará el clustering.

        Returns:
        - Dict[str, List[int]]: Diccionario que contiene la información sobre las clases resultantes del clustering.
        """
        filas, columnas = data.shape
        vertices = particiones_vertices(columnas, 0, 1, filas)
        c_escogidos = self.seleccion_centros(data, vertices)
        centros = np.array([vertices[value['indice']] for value in c_escogidos.values()])
        classes = {f'cluster {i}': [] for i in range(len(centros))}
        for idx, _ in enumerate(data):
            mv_aux = self.mountain_function(centros, data[[idx]])
            elegido = f'cluster {np.argmax(mv_aux)}'
            classes[elegido].append(idx)
        return classes

class Subtractive:
    def __init__(self, ra: float, metrica: str, epsilon: float = 0.2, args={}) -> None:
        """
        Inicializa la clase Subtractive con el parámetro ra y epsilon (opcional).

        Args:
        - ra (float): Radio para la técnica Subtractive.
        - epsilon (float): Umbral para determinar si se debe agregar un nuevo centro (valor predeterminado: 0.2).
        """
        self.ra = ra
        self.rb = 1.5 * ra
        self.epsilon = epsilon
        self.args = args
        self.distance_method = metricas(metrica)

    def density_function(self, data_a: np.ndarray, data_b: np.ndarray) -> np.ndarray:
        """
        Calcula la función de densidad entre los datos proporcionados.

        Args:
        - data_a (np.ndarray): Datos para los cuales se calculará la función de montaña.
        - data_b (np.ndarray): Datos sobre los cuales se calculará la función de montaña.

        Returns:
        - np.ndarray: Peso asociado a cada dato.
        """
        D_i = np.zeros(len(data_a))
        for idx_xi, xi in enumerate(data_a):
            cont = 0
            for _, xj in enumerate(data_b):
                valor_distancia = self.distance_method(**{**{'ma':np.array([xi]),'mb':np.array([xj])},**self.args})[0,0]
                cont += np.exp(-((valor_distancia ** 2) / ((self.ra / 2) ** 2)))
            D_i[idx_xi] = cont
        return D_i
    
    def selecccion_centros(self, data: np.ndarray) -> Dict[str, List[int]]:
        """
        Encuentra los centros utilizando la técnica Subtractive.

        Parameters:
        - data (np.ndarray): Datos sobre los cuales se realizará el clustering.

        Returns:
        - Dict[str, List[int]]: Diccionario que contiene la información sobre las clases resultantes del clustering.
        """
        D_i = self.density_function(data,data)

        c_escogidos = {}
        idx_c1 = np.argmax(D_i)
        c_escogidos['1'] = {'indice': idx_c1, 'm': D_i[idx_c1]}

        bool = True
        cont = 2

        while bool:
            xc1 = data[idx_c1]
            Dc1 = copy.deepcopy(D_i[idx_c1])

            for idx_xi, xi in enumerate(data):
                valor_distancia = self.distance_method(**{**{'ma':np.array([xi]),'mb':np.array([xc1])},**self.args})[0,0]
                D_i[idx_xi] = D_i[idx_xi] - Dc1 * np.exp(-((valor_distancia ** 2) / ((self.rb / 2) ** 2)))
            idx_c1 = np.argmax(D_i)
            if max(D_i) > self.epsilon * c_escogidos[str(cont - 1)]['m']:
                c_escogidos[str(cont)] = {'indice': idx_c1, 'm': D_i[idx_c1]}
                cont += 1
            elif cont > 15:
                bool = False
            else:
                bool = False
        
        return c_escogidos

    def clustering(self, data: np.ndarray) -> Dict[str, List[int]]:
        """
        Realiza el clustering utilizando las densidades entre los datos.

        Parameters:
        - data (np.ndarray): Datos sobre los cuales se realizará el clustering.

        Returns:
        - Dict[str, List[int]]: Diccionario que contiene la información sobre las clases resultantes del clustering.
        """
        c_escogidos = self.selecccion_centros(data)
        centros = np.array([data[value['indice']] for value in c_escogidos.values()])
        classes = {f'cluster {i}': [] for i in range(len(centros))}
        
        for idx, _ in enumerate(data):
            mv_aux = self.density_function(centros, data[[idx]])
            elegido = f'cluster {np.argmax(mv_aux)}'
            classes[elegido].append(idx)
        return classes

class KmeansClustering():
    """
    Implementación de K-Means para la agrupación de datos en k clusters.

    Attributes:
        k (int): Número de clusters deseado.
        centroides (array): Centroides iniciales de los clusters. Si no se proporcionan, se seleccionan aleatoriamente del conjunto de datos.
    """

    def __init__(self, k: int, metrica: str, args={}, centroides=None) -> None:
        """
        Inicializa una instancia de la clase KmeansClustering.

        Args:
            k (int): Número de clusters deseado.
            metrica (str): Métrica bajo la cual se medirán las distancias
            args (dict): Argumentos extras para las funciones de distancia
            centroides (array): Centroides iniciales de los clusters. Si no se proporcionan, se seleccionan aleatoriamente del conjunto de datos.
        """
        self.k = k
        self.centroides = centroides
        self.args = args
        self.distance_method = metricas(metrica)


    def membership_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de membresía asignando cada punto de datos al cluster más cercano.

        Args:
            data (array): Matriz de datos de forma (n_samples, n_features).

        Returns:
            array: Matriz de membresía de forma (k, n_samples) donde cada columna representa la pertenencia de un punto de datos a un cluster mediante valores binarios (0 o 1).
        """
        filas, _ = data.shape

        if self.centroides is None:
            self.centroides = data[np.random.choice(np.arange(filas), size=self.k, replace=False)]

        U = np.zeros((self.k, filas))
        for idx_p, point in enumerate(data):
            u_ij = np.zeros(self.k)
            for idx_ci, ci in enumerate(self.centroides):
                u_ij[idx_ci] = self.distance_method(**{**{'ma':np.array([point]),'mb':np.array([ci])},**self.args})[0,0]
            U[np.argmin(u_ij), idx_p] = 1

        return U

    def cost_function(self, U: np.ndarray, data: np.ndarray) -> float:
        """
        Calcula la función de costo total para la asignación de clusters dada la matriz de membresía.

        Args:
            U (array): Matriz de membresía de forma (k, n_samples).
            data (array): Matriz de datos de forma (n_samples, n_features).

        Returns:
            float: Valor de la función de costo total.
        """
        J = 0
        for idx_ci, ci in enumerate(self.centroides):
            G_i = np.where(U[idx_ci] == 1)[0]
            J_i = 0
            for idx_p in G_i:
                J_i += self.distance_method(**{**{'ma':np.array([data[idx_p]]),'mb':np.array([ci])},**self.args})[0,0]
            J += J_i

        return J

    def clustering(self, data: np.ndarray, tolerancia: float) -> dict:
        """
        Realiza la agrupación (clustering) de datos utilizando el algoritmo K-Means.

        Args:
            data (array): Matriz de datos de forma (n_samples, n_features).
            tolerancia (float): Valor umbral para la convergencia del algoritmo.

        Returns:
            array: Matriz de membresía final después de la convergencia del algoritmo.
        """
        U = self.membership_matrix(data)
        J = self.cost_function(U, data)

        cont = 0
        while J > tolerancia and cont < 10:
            for idx_ci, _ in enumerate(self.centroides):
                aux = np.where(U[idx_ci] == 1)[0]
                self.centroides[idx_ci] = (1 / np.sum(U[idx_ci])) * np.sum(data[aux], axis=0)

            U = self.membership_matrix(data)
            J = self.cost_function(U, data)
            cont += 1
        
        clusters = {}
        for idx_ci, ci in enumerate(U):
            indices = np.where(ci == 1)[0]
            clusters[f'cluster {idx_ci}'] = indices

        return clusters

class FuzzyCMeans():
    """
    Implementación de Fuzzy C-Means para la agrupación de datos en c clusters.

    Attributes:
        c (int): Número de clusters deseado.
        centros (array): Centros iniciales de los clusters. Se calculan durante el proceso de agrupación.
    """

    def __init__(self, c: int, metrica: str, args = {}) -> None:
        """
        Inicializa una instancia de la clase fuzzy_c_means.

        Args:
            c (int): Número de clusters deseado.
        """
        self.c = c
        self.centros = None
        self.args = args
        self.distance_method = metricas(metrica)
    
    def calculate_centers(self, data: np.ndarray, U: np.ndarray, m: float) -> None:
        """
        Calcula los centros de los clusters basándose en la matriz de membresía.

        Args:
            data (array): Matriz de datos de forma (n_samples, n_features).
            U (array): Matriz de membresía de forma (c, n_samples).
            m (float): Parámetro de ponderación para la difusión de la pertenencia.

        Returns:
            None
        """
        _, columnas = data.shape
        centros = []

        for _, uij in enumerate(U):
            cont = np.zeros(columnas)
            for idx_point, point in enumerate(data):
                cont += point * (uij[idx_point] ** m)
            centros.append(cont / (np.sum(uij ** m)))

        self.centros = np.array(centros)

    def cost_function(self, data: np.ndarray, U: np.ndarray, m: float) -> float:
        """
        Calcula la función de costo total para la asignación de clusters dada la matriz de membresía.

        Args:
            data (array): Matriz de datos de forma (n_samples, n_features).
            U (array): Matriz de membresía de forma (c, n_samples).
            m (float): Parámetro de ponderación para la difusión de la pertenencia.

        Returns:
            float: Valor de la función de costo total.
        """
        J = 0

        for idx_uij, uij in enumerate(U):
            Ji = 0
            for idx_p, point in enumerate(data):
                Ji += uij[idx_p] ** m * self.distance_method(**{**{'ma':np.array([point]),'mb':np.array([self.centros[idx_uij]])},**self.args})[0,0]
            J += Ji

        return J

    def update_U(self, data: np.ndarray, m: float) -> np.ndarray:
        """
        Actualiza la matriz de membresía basándose en los centros actuales de los clusters.

        Args:
            data (array): Matriz de datos de forma (n_samples, n_features).
            m (float): Parámetro de ponderación para la difusión de la pertenencia.

        Returns:
            array: Nueva matriz de membresía de forma (c, n_samples).
        """
        filas, _ = data.shape
        new_U = np.zeros((self.c, filas))

        for idx_ci, ci in enumerate(self.centros):
            for idx_p, point in enumerate(data):
                cont = 0
                for _, ck in enumerate(self.centros):
                    distance_ci = self.distance_method(**{**{'ma':np.array([ci]),'mb':np.array([point])},**self.args})[0,0]
                    distance_ck = self.distance_method(**{**{'ma':np.array([ck]),'mb':np.array([point])},**self.args})[0,0]
                    cont += (distance_ci / distance_ck) ** (2 / (m - 1))
                new_U[idx_ci, idx_p] = 1 / cont

        return new_U
    
    def clustering(self, data: np.ndarray, m: float, tolerancia: float) -> np.ndarray:
        """
        Realiza la agrupación (clustering) de datos utilizando el algoritmo Fuzzy C-Means.

        Args:
            data (array): Matriz de datos de forma (n_samples, n_features).
            m (float): Parámetro de ponderación para la difusión de la pertenencia.
            tolerancia (float): Valor umbral para la convergencia del algoritmo.

        Returns:
            array: Matriz de membresía final después de la convergencia del algoritmo.
        """
        filas, _ = data.shape
        U = np.random.rand(self.c, filas)
        U /= U.sum(axis=0)

        self.calculate_centers(data, U, m)
        J = self.cost_function(data, U, m)

        cont = 0
        while J > tolerancia and cont < 10:
            U = self.update_U(data, m)
            self.calculate_centers(data, U, m)
            J = self.cost_function(data, U, m)
            cont += 1

        clusters = {f'cluster {i}': [] for i in range(U.shape[0])}
        for idx_p, _ in enumerate(data):
            aux = np.argmax(U[:, idx_p])
            clusters[f'cluster {aux}'].append(idx_p)

        return clusters

class SpectralClustering():
    """
    Implementación de Spectral Clustering para la agrupación de datos en k clusters.

    Attributes:
        k (int): Número de clusters deseado.
    """

    def __init__(self, k: int, metrica: str, args = {}, centros = None) -> None:
        """
        Inicializa una instancia de la clase SpectralClustering.

        Args:
            k (int): Número de clusters deseado.
            metrica (str): Métrica bajo la cual se medirán las distancias
            args (dict): Argumentos extras para las funciones de distancia
            centros (None | array): Centros de los clusters 
        """
        self.k = k
        self.args = args
        self.metrica = metrica
        self.centros = centros

    def clustering(self, sigma: float, distances: np.ndarray ) -> dict:
        """
        Realiza la agrupación (clustering) de datos utilizando el algoritmo Spectral Clustering.

        Args:
            sigma (float): Parámetro para el cálculo de la similitud entre los puntos de datos.
            distances (array): Matriz de distancias de los datos

        Returns:
            array: Matriz de membresía final después de la convergencia del algoritmo.
        """

        similarity_matrix = np.exp(-distances ** 2 / (2.0 * (sigma ** 2)))
        diagonal_matrix = np.diag(similarity_matrix.sum(axis=1))

        laplacian_matrix = diagonal_matrix - similarity_matrix
        eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)

        eigen_indices = np.argsort(eigenvalues)[:self.k]
        eigenvectors = eigenvectors[:, eigen_indices]
        transformed_data = eigenvectors

        instance = KmeansClustering(self.k,self.metrica,self.args, self.centros)
        clusters = instance.clustering(transformed_data, 5)

        return clusters