import os
from model import load_model, Agent
from read_esco import read_dataset
from utils import print_colored
from datasets import Dataset, load_dataset, concatenate_datasets
import pandas as pd

def main():

    # Config
    overwrite_mode = True
    max_job_per_occupation = 3
    similarity_threshold = 0.8
    data_path = os.path.dirname(os.path.abspath(__file__)) # TODO da standardizzare
    model_path = "E:/text-generation-webui-main/models/saiga-7b"

    # load dataset from huggingface
    existing_dataset = load_dataset('FinancialSupport/SynthEscoJobAds')

    occupations = read_dataset(data_path)
    model, tokenizer = load_model(model_path)

    # Initialize agent
    Writer = Agent(model, tokenizer, 'cuda') #create job description
    Supervisor = Agent(model, tokenizer, 'cuda') #quality check of the job description

    # Iterate over occupations
    for i, row in occupations.iterrows():

        #Check if 'preferredLabel' is already in the dataset
        if (existing_dataset.filter(lambda example: example['escoLabel'] == row['preferredLabel']).num_rows['train'] > 0) or (overwrite_mode == False): 
            print_colored(f"[Boss]: {row['preferredLabel']} already in the dataset. Proceed with the next topic", "red")
            continue

        # Reset agents memory
        Writer.forget()
        Supervisor.forget()
        j = 0

        print_colored(f"[Boss]: incoming task #{i+1} for the team", "red")

        # Construct task string
        generate_task = f"Generate a realistic job description for the role of {row['preferredLabel']},\
                \nwe are talking about a {row['description']}. It is also known as:\n{row['altLabels']}."

        print_colored(f"[Writer]: receiving task #{i+1}:", "blue")
        print_colored(generate_task, "blue")

        # Generate job descriptions
        answers = Writer.answer(generate_task, batch_generate = max_job_per_occupation)
        print_colored(f"\n[Writer]: jd creation task #{i+1} executed {max_job_per_occupation} times", "green")
        #print_colored('\n\n@@END@@\n\n'.join(answers), "green")

        # Save job descriptions in memory of agents 2
        Writer.memorize(answers)
        Supervisor.memorize(Writer.remember())

        # Iterate over job descriptions
        while True:

            # Check if job descriptions are too similar
            jd_too_similar = Supervisor.find_similarity_in_memory(similarity_threshold)
            if not jd_too_similar:
                break 
            jd_pair_too_similar = jd_too_similar[0] 

            # Construct task string
            supervise_task = f"Repharse this job description: {answers[jd_pair_too_similar[1]]},\
            \nbecause it is too similar to this other job description: {answers[jd_pair_too_similar[2]]}.\
            \nRememebre that we are talking about a {row['description']}. It is also known as:\n{row['altLabels']}."
            print_colored(f"[Supervisor]: Found two similar job description in task #{i+1} ({jd_pair_too_similar[0]*100}%).", "blue")
            #print_colored(supervise_task, "blue")

            fix = Supervisor.answer(supervise_task, batch_generate=1)
            print_colored(f"\n[Supervisor]: task #{j+1} fixed", "green")
            #print_colored(fix, "green")

            # Refresh memory of agents 2
            old_memory = Supervisor.remember()
            new_memory = old_memory
            new_memory[jd_pair_too_similar[1]] = fix
            Supervisor.memorize(new_memory)

            # Update counter
            j += 1
        
        # Save memory of agent 2 in a huggingface dataset
        final_memory = Supervisor.remember()
        # Add 'preferredLabel' to each job ad
        final_memory_dicts = [{'job_ad': job_ad, 'escoLabel': row['preferredLabel']} for job_ad in final_memory]
        # Convert the list of memories into a Hugging Face Dataset
        new_data = Dataset.from_pandas(pd.DataFrame(final_memory_dicts))
        # Concatenate the existing dataset with the new data
        updated_dataset = concatenate_datasets([existing_dataset['train'], new_data])
        # Save the dataset
        if overwrite_mode:
            new_data.push_to_hub('FinancialSupport/SynthEscoJobAds')
        else:
            updated_dataset.push_to_hub('FinancialSupport/SynthEscoJobAds')

        break

if __name__ == "__main__":
    main()