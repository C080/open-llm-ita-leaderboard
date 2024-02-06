import gradio as gr
import pandas as pd

csv_filename = 'leaderboard.csv'
url = 'https://docs.google.com/spreadsheets/d/1Oh3nrbdWjKuh9twJsc9yJLppiJeD_BZyKgCTOxRkALM/export?format=csv'

def get_data():
    return pd.read_csv(url)

with gr.Blocks() as demo:
        with gr.Tab('Classifica'):

            gr.Markdown("# Classifica degli LLM italiani")
            form_link = "https://forms.gle/Gc9Dfu52xSBhQPpAA"
            gr.Markdown(f"I modelli sono testati su SQuAD-it e ordinati per F1 Score e EM (Exact Match).Si ringrazia il @galatolo per il codice dell'eval. Se volete aggiungere il vostro modello compilate il form {form_link}")
            gr.Dataframe(pd.read_csv(csv_filename, sep=';'))

        with gr.Tab('Test della community'):
            gr.Markdown("# Evaluation aggiuntive fatte dalla community")
            discord_link = 'https://discord.com/invite/nfgaTG3H'
            gr.Markdown(f"@giux78 sta lavorando sull'integrazione di nuovi dataset di benchmark italiani. Se volete contribuire anche voi unitevi al discord della community {discord_link}")
            gr.DataFrame(get_data, every=3600)
            
demo.launch()