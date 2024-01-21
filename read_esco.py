import pandas as pd



def read_and_prepare_datasets(path):

    # Import
    df_occupations = pd.read_csv(path + "\data\occupations_en.csv")
    df_skills = pd.read_csv(path + "\data\skills_en.csv")
    df_bridge = pd.read_parquet(path + "\data\occupation_has_skill.parquet")

    # Prepare
    df_occupations = df_occupations[df_occupations.code.str.replace(".","").str.len() == 5] # filter occupation_code with only 5 digit
    df_occupations = df_occupations[["preferredLabel", "altLabels", "description", "conceptUri"]]
    

    # Linking skill to occupation with bridge table
    df_occupations['conceptUri'] = df_occupations['conceptUri'].str.split('/').str[-1]
    df_skills['conceptUri'] = df_skills['conceptUri'].str.split('/').str[-1]
    df_bridge["occ_id"] = df_bridge["occ_id"].str.split("_").str[1]
    df_bridge["skill_id"] = df_bridge["skill_id"].str.split("_").str[1]
    # use bridge table to link skill to occupation
    df_occupations = df_occupations.merge(df_bridge, left_on="conceptUri", right_on="occ_id")
    df_occupations = df_occupations.merge(df_skills, left_on="skill_id", right_on="conceptUri")


    
    


    return df_occupations
