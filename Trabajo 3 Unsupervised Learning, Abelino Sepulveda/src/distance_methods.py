import numpy as np
from src.auxiliar_methods import calculate_covariance
from typing import Union, Callable

def distancia_euclidea(ma: np.ndarray, mb: np.ndarray = None) -> np.ndarray:
    """
    Calcula la distancia euclidiana entre las filas de las matrices ma y mb.

    Parameters:
    - ma (np.ndarray): Matriz de datos.
    - mb (np.ndarray, optional): Otra matriz de datos. Si no se proporciona, se asume que es igual a ma.

    Returns:
    - np.ndarray: Matriz de distancias euclidianas entre las filas de ma y mb.
    """
    if mb is None:
        mb = ma
    filas_ma, _ = ma.shape
    filas_mb, _ = mb.shape
    distancias = np.zeros((filas_ma,filas_mb))
    for idx_ma, dato_ma in enumerate(ma):
        for idx_mb, dato_mb in enumerate(mb):
            distancias[idx_ma,idx_mb] =  np.sqrt(np.sum((dato_ma-dato_mb)**2))
    return distancias

def distancia_manhattan(ma: np.ndarray, mb: np.ndarray = None) -> np.ndarray:
    """
    Calcula la distancia de Manhattan entre las filas de las matrices ma y mb.

    Parameters:
    - ma (np.ndarray): Matriz de datos.
    - mb (np.ndarray, optional): Otra matriz de datos. Si no se proporciona, se asume que es igual a ma.

    Returns:
    - np.ndarray: Matriz de distancias de Manhattan entre las filas de ma y mb.
    """
    if mb is None:
        mb = ma
    filas_ma, _ = ma.shape
    filas_mb, _ = mb.shape
    distancias = np.zeros((filas_ma,filas_mb))
    for idx_ma, dato_ma in enumerate(ma):
        for idx_mb, dato_mb in enumerate(mb):
            distancias[idx_ma,idx_mb] = np.sum(abs(dato_ma-dato_mb))
    return distancias

def distancia_mahalanobis(ma: np.ndarray, mb: np.ndarray = None) -> np.ndarray:
    """
    Calcula la distancia de Mahalanobis entre las filas de las matrices ma y mb.

    Parameters:
    - ma (np.ndarray): Matriz de datos.
    - mb (np.ndarray, optional): Otra matriz de datos. Si no se proporciona, se asume que es igual a ma.

    Returns:
    - np.ndarray: Matriz de distancias de Mahalanobis entre las filas de ma y mb.
    """
    if mb is None:
        mb = ma
    filas_ma, _ = ma.shape
    filas_mb, _ = mb.shape
    distancias = np.zeros((filas_ma,filas_mb))

    covarianzas = calculate_covariance(ma.T,mb.T)
    inverse_cov = np.linalg.inv(covarianzas)

    for i in range(filas_ma):
        for j in range(filas_mb):
            print(np.dot((ma[i,:]-mb[j,:]).reshape(1,-1),np.dot(inverse_cov,(ma[i,:]-mb[j,:]).reshape(-1,1))))
            distancias[i,j] = np.dot((ma[i,:]-mb[j,:]).reshape(1,-1),np.dot(inverse_cov,(ma[i,:]-mb[j,:]).reshape(-1,1)))
    
    return distancias

def norma_lp(ma: np.ndarray, mb: np.ndarray = None, p: Union[int, float, str] = 'infinito') -> np.ndarray:
    """
    Calcula la norma Lp entre las filas de las matrices ma y mb.

    Parameters:
    - ma (np.ndarray): Matriz de datos.
    - mb (np.ndarray, optional): Otra matriz de datos. Si no se proporciona, se asume que es igual a ma.
    - p (Union[int, float, str]): Parámetro que indica el tipo de norma. Puede ser un número entero o de punto flotante,
      o la cadena "infinito" para la norma infinita.

    Returns:
    - np.ndarray: Matriz de normas Lp entre las filas de ma y mb.
    """
    if mb is None: 
        mb = ma
    filas_ma, _ = ma.shape
    filas_mb, _ = mb.shape
    distancias = np.zeros((filas_ma,filas_mb))

    for i in range(filas_ma):
        for j in range(filas_mb):
            if p=="infinito":
                distancias[i,j] = np.max(abs(ma[i,:]-mb[j,:]))
            else:
                distancias[i,j] = np.sum(abs(ma[i,:]-mb[j,:])**p)**(1/p)
    return distancias

def distancia_coseno(ma: np.ndarray, mb: np.ndarray = None) -> np.ndarray:
    """
    Calcula la distancia coseno entre las filas de las matrices ma y mb.

    Parameters:
    - ma (np.ndarray): Matriz de datos.
    - mb (np.ndarray, optional): Otra matriz de datos. Si no se proporciona, se asume que es igual a ma.

    Returns:
    - np.ndarray: Matriz de distancias coseno entre las filas de ma y mb.
    """
    if mb is None:
        mb = ma
    filas_ma, _ = ma.shape
    filas_mb, _ = mb.shape
    distancias = np.zeros((filas_ma, filas_mb))
    for i in range(filas_ma):
        for j in range(filas_mb):
            numerador = np.sum(ma[i, :] * mb[j, :])
            norma_ma = np.sqrt(np.sum(ma[i, :]**2))
            norma_mb = np.sqrt(np.sum(mb[j, :]**2))
            
            # Comprobación para evitar la división por cero
            if norma_ma == 0 or norma_mb == 0:
                distancias[i, j] = 0
            else:
                distancias[i, j] = 1 - numerador / (norma_ma * norma_mb)
    return distancias

def metricas(metrica: str) -> Callable:
    """
    Selecciona y devuelve la función de distancia correspondiente según la métrica especificada.

    Parameters:
    - metrica (str): Nombre de la métrica. Puede ser 'euclidea', 'manhattan', 'mahalanobis', 'lp', o 'cosine'.

    Returns:
    - Callable: Función de distancia correspondiente.
    """
    metodo_distancia: Callable

    # Seleccionar la función de distancia según la métrica
    if metrica == 'euclidea':
        metodo_distancia = distancia_euclidea
    elif metrica == 'manhattan':
        metodo_distancia = distancia_manhattan
    elif metrica == 'mahalanobis':
        metodo_distancia = distancia_mahalanobis
    elif metrica == 'lp':
        metodo_distancia = norma_lp
    elif metrica == 'cosine':
        metodo_distancia = distancia_coseno
    else:
        raise ValueError(f"Métrica no válida: {metrica}")

    return metodo_distancia