# importe de las librerias 
import json
from src.auxiliar_methods import normalize_min_max, plotting_clusters, create_folder_if_not_exists,plotting_clusters, convert_to_serializable
from src.clustering_methods import Mountain, Subtractive, separate_boxes_by_datapoint, separate_boxes, KmeansClustering, FuzzyCMeans, SpectralClustering
from src.distance_methods import metricas
from src.eda_data import estadisticas_de_los_datos
from src.umap_embedding import umap_embedding
from src.indices import intra_cluster, extra_cluster
from src.encoder import autoencoder_dimension_reduction_ampliation
import pandas as pd

def clustering_metodos(data,folder_path, args_metodo):
    print(args_metodo['espacio'],args_metodo['data_name'])
    indices_clusters = {}
    metricas_ = ['euclidea', 'manhattan','lp' ,'cosine']
    for metrica in metricas_:
        matriz_distancia = metricas(metrica)(data)
        args = {}
        if metrica == 'lp':
            args = {'p': 'infinito'}
        for alfa in [0.3, 0.43,0.8]:
            mountain_instance = Mountain(alfa, metrica,args)
            mountain_clusters = mountain_instance.clustering(data)
            plotting_clusters(mountain_clusters,data,folder_path,{'metodo':'mountain','metrica':metrica,'alpha':alfa, 'espacio': args_metodo['espacio'], 'data_name': args_metodo['data_name']})
            indices_clusters[f'mountain {metrica} alfa = {alfa}'] = {
                'intra' : intra_cluster(data, mountain_clusters),
                'extra' : extra_cluster(data,mountain_clusters)
            }

            numero_de_clusters = len(mountain_clusters.keys())
            clusters_de_juan = separate_boxes_by_datapoint(matriz_distancia, numero_de_clusters)
            plotting_clusters(clusters_de_juan,data,folder_path,{'metodo':'cajar_de_juan','metrica':metrica,'num_clusters':numero_de_clusters,'alpha':alfa, 'espacio': args_metodo['espacio'], 'data_name': args_metodo['data_name']})
            indices_clusters[f'cajas de juan {metrica} n_clusters = {numero_de_clusters}'] = {
                'intra' : intra_cluster(data, clusters_de_juan),
                'extra' : extra_cluster(data,clusters_de_juan)
            }

            clusters_de_sofia = separate_boxes(matriz_distancia,numero_de_clusters)
            plotting_clusters(clusters_de_sofia,data,folder_path,{'metodo':'vecinos_de_sofia','metrica':metrica,'num_clusters':numero_de_clusters,'alpha':alfa, 'espacio': args_metodo['espacio'], 'data_name': args_metodo['data_name']})
            indices_clusters[f'vecinos de sofia {metrica} n_clusters = {numero_de_clusters} alpha={alfa}'] = {
                'intra' : intra_cluster(data, clusters_de_sofia),
                'extra' : extra_cluster(data,clusters_de_sofia)
            }

            k_means_instance = KmeansClustering(numero_de_clusters,metrica,args)
            kmeans_clusters = k_means_instance.clustering(data,0.1)
            plotting_clusters(kmeans_clusters,data,folder_path,{'metodo':'kmeans','metrica':metrica,'num_clusters':numero_de_clusters,'alpha':alfa, 'espacio': args_metodo['espacio'], 'data_name': args_metodo['data_name']})
            indices_clusters[f'k means {metrica} n_clusters = {numero_de_clusters} alpha={alfa}'] = {
                'intra' : intra_cluster(data, kmeans_clusters),
                'extra' : extra_cluster(data,kmeans_clusters)
            }

            fuzzy_instance = FuzzyCMeans(numero_de_clusters,metrica,args)
            fuzzy_clusters = fuzzy_instance.clustering(data,2,0.1)
            plotting_clusters(fuzzy_clusters,data,folder_path,{'metodo':'fuzzy','metrica':metrica,'num_clusters':numero_de_clusters,'alpha':alfa, 'espacio': args_metodo['espacio'], 'data_name': args_metodo['data_name']})
            indices_clusters[f'fuzzy c means {metrica} n_clusters = {numero_de_clusters} alpha={alfa}'] = {
                'intra' : intra_cluster(data, fuzzy_clusters),
                'extra' : extra_cluster(data,fuzzy_clusters)
            }
            
            spectral_c_instance = SpectralClustering(numero_de_clusters,metrica, args)
            spectral_centros = spectral_c_instance.clustering(0.3,matriz_distancia)
            plotting_clusters(spectral_centros,data,folder_path,{'metodo':'spectral','metrica':metrica,'num_clusters':numero_de_clusters,'alpha':alfa, 'espacio': args_metodo['espacio'], 'data_name': args_metodo['data_name']})
            indices_clusters[f'spectral {metrica} n_clusters = {numero_de_clusters} alpha={alfa}'] = {
                'intra' : intra_cluster(data, spectral_centros),
                'extra' : extra_cluster(data,spectral_centros)
            }

        for ra in [0.1,0.5,0.9]:
            subtractive_intance = Subtractive(ra,metrica,args=args)
            subtractive_cluster = subtractive_intance.clustering(data)
            plotting_clusters(subtractive_cluster,data,folder_path,{'metodo':'subtractive','metrica':metrica,'ra':ra, 'espacio': args_metodo['espacio'], 'data_name': args_metodo['data_name']})
            indices_clusters[f'subtractive {metrica} n_clusters = {numero_de_clusters} ra={ra}'] = {
                'intra' : intra_cluster(data, subtractive_cluster),
                'extra' : extra_cluster(data,subtractive_cluster)
            }

        with open(f"{folder_path}/indices_{args_metodo['espacio']}_{args_metodo['data_name']}.json", 'w') as archivo_json:
            json.dump(convert_to_serializable(indices_clusters), archivo_json)

def experimentacion(path_df,args_metodo):
    df = pd.read_csv(path_df).dropna()
    columns_to_normalize = df.columns[:-1]
    df[columns_to_normalize] = normalize_min_max(df[columns_to_normalize].values)
    data = df[columns_to_normalize].values

    ## EDA 
    # estadisticas_de_los_datos(df,args_metodo)

    # espacio normal 
    clustering_metodos(data,args_metodo['folder_path'], {'espacio':'normal', 'data_name':args_metodo['data_name'] })

    # espacio aumentado
    print('aumentando')
    aumented_space = autoencoder_dimension_reduction_ampliation(data,False)
    clustering_metodos(aumented_space,args_metodo['folder_path'], {'espacio':'aumentado', 'data_name': args_metodo['data_name']})

    ## espacio de embebimiento
    embedding = umap_embedding(data,df,args_metodo)
    clustering_metodos(embedding,args_metodo['folder_path'], {'espacio':'embedding', 'data_name':args_metodo['data_name']})

def main(folder_path):
    dfs_paths = [ '/Users/asepulvede/Desktop/Universidad/Noveno Semestre/Inteligencia Artificial/Trabajo 3 Unsupervised Learning, Abelino Sepulveda/datasets/iris.csv',
        '/Users/asepulvede/Desktop/Universidad/Noveno Semestre/Inteligencia Artificial/Trabajo 3 Unsupervised Learning, Abelino Sepulveda/datasets/wheat_seeds.csv'
    ]

    data_names = ['iris', 'wheat_seeds']
    target_column_names = ['species', 'Class']
    
    # dfs_paths = [ '/Users/asepulvede/Desktop/Universidad/Noveno Semestre/Inteligencia Artificial/Trabajo 3 Unsupervised Learning, Abelino Sepulveda/datasets/wheat_seeds.csv'
    # ]

    # data_names = ['wheat_seeds']
    # target_column_names = ['Class']
    
    for idx,dfs_path in enumerate(dfs_paths) :
        args_metodo = {
            'folder_path': folder_path,
            'data_name': data_names[idx],
            'target_column_name': target_column_names[idx]
        }
        experimentacion(dfs_path, args_metodo)


if __name__ == "__main__":
    folder_path = 'results'
    create_folder_if_not_exists(folder_path)
    main(folder_path)

    