import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def estadisticas_de_los_datos(df, args):
    # pairplot
    sns.pairplot(df, hue=args['target_column_name'])
    plt.savefig(f"{args['folder_path']}/pairplot_{args['data_name']}.png")
    plt.show()


    # pairgrid
    g = sns.PairGrid(df, diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    plt.savefig(f"{args['folder_path']}/pairgrid_{args['data_name']}.png")
    plt.show()

    # boxplots
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Box(y=df[col], name=col))
    fig.write_image(f"{args['folder_path']}/boxplot_{args['data_name']}.png")
    fig.show()

    # estadisticas descriptivas
    descriptive_st = df.describe()
    descriptive_st.to_csv(f"{args['folder_path']}/descriptive_statistics_{args['data_name']}.csv")