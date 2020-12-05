#!/usr/bin/env python

import re
import pandas as pd

def get_user_table():
    """
    return user table
    """

    df = pd.read_csv("./data/lessons.csv")
    return([[i+1, df["lesson"][i], ", ".join(re.split(";",df["topics"][i])), df["repo"][i], df["checkpoint"][i]] for i in df.index])

if __name__ == "__main__":
    
    table = get_user_table()

    for row in table:
        print(row)
