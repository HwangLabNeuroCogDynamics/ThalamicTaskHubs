import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def loadings_heatmap(csv_path):
    df = pd.read_csv(csv_path).set_index('Task')
    ax = sns.heatmap(df)
    plt.xlabel('PCA Component', fontsize=15)
    plt.ylabel('Task', fontsize=15)
    plt.show()

loadings_heatmap('/mnt/nfs/lss/lss_kahwang_hpc/data/MDTB/analysis/_blockcorrelated_loadings.csv')
