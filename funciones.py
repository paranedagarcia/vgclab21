import pandas as pd
import numpy as np


def load_data_csv(filename, chunks=False):
    # Load the data from the CSV file separated by comma or semicomma

    if filename is not None:
        try:
            data = pd.read_csv(filename, sep=",",
                               low_memory=False, index_col=0)
        except:
            data = pd.read_csv(filename, sep=";",
                               low_memory=False, index_col=0)
        return data
