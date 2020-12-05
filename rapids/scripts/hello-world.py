import os
import time
import cudf
import pandas as pd

def print_time(start):
    """convenience function to print runtime"""
    m, s = divmod(time.time() - start, 60)
    h, m = divmod(m, 60)
    print(f"{int(h):d}:{int(m):02d}:{int(s):02d}")

# load data as you normally would into pandas and print some means
data_dir = os.path.join("..","data")
file_path = os.path.join(data_dir, "US_Accidents_June20.csv")
numeric_cols = ["Distance(mi)", "Precipitation(in)", "Temperature(F)", "Wind_Speed(mph)",
                "Severity"]

start = time.time()
df = pd.read_csv(file_path)
print(df.loc[:, numeric_cols].mean())
print_time(start)

# load data with cuDF and print some means
start = time.time()
gdf = cudf.read_csv(os.path.join(data_dir, "US_Accidents_June20.csv"))
for column in numeric_cols:
    print(column, round(gdf[column].mean(), 2))

print_time(start)