import os
from model import load_model, Agent
from read_esco import read_dataset
from utils import print_colored
from datasets import Dataset, load_from_disk, concatenate_datasets
import pandas as pd

def main():

    # Config
    max_job_per_occupation = 3
    similarity_threshold = 0.8
    data_path = os.path.dirname(os.path.abspath(__file__)) # TODO da standardizzare
    model_path = "E:/text-generation-webui-main/models/saiga-7b"
    output_path = "E:/uni/thesis/output"

    # Check if the dataset file exists
    if os.path.exists(output_path):
        # Load existing dataset
        existing_dataset = load_from_disk(output_path)
    else:
        # Initialize an empty dataset
        existing_dataset = Dataset.from_pandas(pd.DataFrame(columns=['memory']))

    occupations = read_dataset(data_path)
    model, tokenizer = load_model(model_path)

    # Initialize agent
    agent_1 = Agent(model, tokenizer, 'cuda') #create job description
    agent_2 = Agent(model, tokenizer, 'cuda') #quality check of the job description

    # Iterate over occupations
    for i, row in occupations.iterrows():

        # Reset agents memory
        agent_1.forget()
        agent_2.forget()
        j = 0

        print_colored(f"Boss: incoming task #{i+1} for Agent 1", "red")

        # Construct task string
        generate_task = f"Generate a realistic job description for the role of {row['preferredLabel']},\
                \nwe are talking about a {row['description']}. It is also known as:\n{row['altLabels']}."

        print_colored(f"Agent 1: receiving task #{i+1}:", "blue")
        print_colored(generate_task, "blue")

        # Generate job descriptions
        answers = agent_1.answer(generate_task, batch_generate = max_job_per_occupation)
        print_colored(f"\nAgent 1: task #{i+1} executed {max_job_per_occupation} times", "green")
        #print_colored('\n\n@@END@@\n\n'.join(answers), "green")

        # Save job descriptions in memory of agents 2
        agent_1.memorize(answers)
        agent_2.memorize(agent_1.remember())

        # Iterate over job descriptions
        while True:

            # Check if job descriptions are too similar
            jd_too_similar = agent_2.find_similarity_in_memory(similarity_threshold)
            if not jd_too_similar:
                break 
            jd_pair_too_similar = jd_too_similar[0] 

            print_colored(f"Boss: incoming task #{j+1} for Agent 2", "red")
            # Construct task string
            supervise_task = f"Repharse this job description: {answers[jd_pair_too_similar[1]]},\
            \nbecause it is too similar to this other job description: {answers[jd_pair_too_similar[2]]}.\
            \nRememebre that we are talking about a {row['description']}. It is also known as:\n{row['altLabels']}."
            print_colored(f"Agent 2: supervising task #{i+1}. Found two job description {jd_pair_too_similar[0]*100}% similar.", "blue")
            #print_colored(supervise_task, "blue")

            fix = agent_2.answer(supervise_task, batch_generate=1)
            print_colored(f"\nAgent 2: task #{j+1} fixed", "green")
            #print_colored(fix, "green")

            # Refresh memory of agents 2
            old_memory = agent_2.remember()
            new_memory = old_memory
            new_memory[jd_pair_too_similar[1]] = fix
            agent_2.memorize(new_memory)

            # Update counter
            j += 1
        
        # Save memory of agent 2 in a huggingface dataset
        final_memory = agent_2.remember()
        # Convert the list of memories into a Hugging Face Dataset
        new_data = Dataset.from_pandas(pd.DataFrame(final_memory, columns=['memory']))
        # Concatenate the existing dataset with the new data
        updated_dataset = concatenate_datasets([existing_dataset, new_data])
        # Save the dataset
        updated_dataset.save_to_disk(output_path)

        break

if __name__ == "__main__":
    main()