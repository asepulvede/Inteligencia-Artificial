import umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def umap_embedding(data_matrix: np.ndarray, df: pd.DataFrame, args: dict) -> np.ndarray:
    """
    Realiza un embebimiento UMAP y visualiza la proyección en un gráfico de dispersión.

    Args:
        data_matrix (array): Matriz de datos de alta dimensión de forma (n_samples, n_features).
        df (DataFrame): DataFrame que contiene los datos, especialmente la columna de destino para la coloración.
        args (dict): Configuración adicional para la función, incluyendo 'target_column_name', 'data_name' y 'path'.

    Returns:
        array: La representación de baja dimensión obtenida mediante UMAP.
    """

    embedding = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42).fit_transform(data_matrix)

    categories = df[args['target_column_name']].unique()
    category_to_number = {category: number for number, category in enumerate(categories)}

    plt.scatter(embedding[:, 0], embedding[:, 1],c=[sns.color_palette()[x] for x in df[args['target_column_name']].map(category_to_number)])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f"UMAP projection of the {args['data_name']} dataset", fontsize=24)
    plt.savefig(f"{args['folder_path']}/embedding_{args['data_name']}.png")
    plt.show()

    return embedding




