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
            gr.Markdown(f"Nella tabella la classifica dei risultati ottenuti confrontando alcuni modelli LLM italiani utilizzando questa [repo github](https://github.com/C080/open-llm-ita-leaderboard) da me mantenuta. I modelli sono testati su SQuAD-it e ordinati per F1 Score e EM (Exact Match). Si ringrazia il @galatolo per il codice dell'eval. Se volete aggiungere il vostro modello compilate il form {form_link}.")
            gr.Dataframe(pd.read_csv(csv_filename, sep=';'))

            gr.Markdown('''# Community discord
            Se volete contribuire o semplicemente partecipare unitevi al nostro [discord](https://discord.com/invite/nfgaTG3H) per rimanere aggiornati su LLM in lingua italiana. 

            # Sponsor
            Le evaluation sono state sponsorizzate da un provider cloud italano [seeweb.it](https://www.seeweb.it/) molto attento al mondo dell'AI e con un ottima offerta di GPUs ed esperienza di sviluppo.


            # NON E' una classifica ma una evaluation 

            In questa tabella una serie di evaluations create con [lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness) e sponsorizzate da un cloud provider italiano [seeweb](https://www.seeweb.it/)  su tasks appositi per l'italiano. Abbiamo anche contribuito con questa [PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1358) in attesa di essere mergiata aggiungendo il task per multilingual mmul e contiamo di migliorare gli eval sull'italiano con altre PR.

            Dopo aver installato [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) per generare i risultati i comandi:

                lm_eval --model hf --model_args pretrained=HUGGINGFACE_MODEL_ID  --tasks xcopa_it,hellaswag_it,lambada_openai_mt_it,belebele_ita_Latn,m_mmlu_it  --device cuda:0 --batch_size 8 

            oppure per few shot 3:
 
            lm_eval --model hf --model_args pretrained=HUGGINGFACE_MODEL_ID  --tasks m_mmlu_it  --num_fewshot 3 --device cuda:0 --batch_size 8  
            ''')
            discord_link = 'https://discord.com/invite/nfgaTG3H'
            gr.Markdown(f"@giux78 sta lavorando sull'integrazione di nuovi dataset di benchmark italiani. Se volete contribuire anche voi unitevi al discord della community {discord_link}")
            gr.DataFrame(get_data, every=3600)
            
demo.launch()