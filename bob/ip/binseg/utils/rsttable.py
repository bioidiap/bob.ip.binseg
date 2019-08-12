from bob.ip.binseg.utils.pdfcreator import get_paths
import pandas as pd
from tabulate import tabulate
import os

def create_overview_grid(output_path):
    """ Reads all Metrics.csv in a certain output path and pivots them to a rst grid table"""
    filename = 'Metrics.csv'
    metrics = get_paths(output_path,filename)
    f1s = []
    stds = []
    models = []
    databases = []
    for m in metrics:
        metrics = pd.read_csv(m)
        maxf1 = metrics['f1_score'].max()
        idmaxf1 = metrics['f1_score'].idxmax()
        std = metrics['std_f1'][idmaxf1]
        stds.append(std)
        f1s.append(maxf1)
        model = m.split('/')[-3]
        models.append(model)
        database = m.split('/')[-4]
        databases.append(database)
    df = pd.DataFrame()
    df['database'] = databases
    df['model'] = models
    df['f1'] = f1s
    df['std'] = stds
    pivot = df.pivot(index='database',columns='model',values='f1')
    pivot2 = df.pivot(index='database',columns='model',values='std')

    with open (os.path.join(output_path,'Metrics_overview.rst'), "w+") as outfile:
        outfile.write(tabulate(pivot,headers=pivot.columns, tablefmt="grid"))
    with open (os.path.join(output_path,'Metrics_overview_std.rst'), "w+") as outfile:
        outfile.write(tabulate(pivot2,headers=pivot2.columns, tablefmt="grid"))