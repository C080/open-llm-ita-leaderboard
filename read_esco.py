import pandas as pd



def read_dataset(path):
    df_occupations = pd.read_csv(path + "\data\occupations_en.csv")
    df_occupations_filtered = df_occupations[["preferredLabel", "altLabels", "description"]]
    #print(f"Read {len(df_occupations_filtered)} occupations.")
    return df_occupations_filtered
