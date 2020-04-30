import pandas as pd
import numpy as np


def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v


df = pd.read_excel("preliminary.xlsx", sheet_name="Responses")
catA = df[[2, 7, 15, 9, 17, 16, 23, 6, 11, 14, 13]].copy()

catA.describe