import os
from model import load_model, Agent
from read_esco import read_dataset
from utils import print_colored

def main():

    # Config
    max_job_per_occupation = 3
    similarity_threshold = 0.4
    data_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "E:/text-generation-webui-main/models/saiga-7b"

    occupations = read_dataset(data_path)
    model, tokenizer = load_model(model_path)

    # Initialize agent
    agent_1 = Agent(model, tokenizer, 'cuda') #create job description
    agent_2 = Agent(model, tokenizer, 'cuda') #quality check of the job description

    # Iterate over occupations
    for i, row in occupations.iterrows():
        print_colored(f"Boss: incoming task #{i+1} for Agent 1", "red")

        # Construct task string
        generate_task = f"Generate a realistic job description for the role of {row['preferredLabel']},\
                \nwe are talking about a {row['description']}. It is also known as:\n{row['altLabels']}."

        print_colored(f"Agent 1: receiving task #{i+1}:", "blue")
        print_colored(generate_task, "blue")

        # Generate job descriptions
        answers = agent_1.answer(generate_task, batch_generate = max_job_per_occupation)
        print_colored(f"\nAgent 1: task #{i+1} executed {max_job_per_occupation} times)", "green")
        print_colored(answers, "green")

        # Save job descriptions in memory of agents 2
        agent_1.memorize(answers)
        agent_2.memorize(agent_1.remeber())

        # Check if job descriptions are too similar
        most_similar_jd = agent_2.find_similarity_in_memory(similarity_threshold)

        while most_similar_jd != []:

            for jd_pair_too_similar in most_similar_jd:
                # Construct task string
                supervise_task = f"Read the first job description: {answers[jd_pair_too_similar[1]]}.\
                \nNow read this second job description: {answers[jd_pair_too_similar[2]]}.\
                \nThey refer to the same job, but are these two job descriptions written almost the same? Answer Yes or no."
                print_colored(f"Agent 2: supervising task #{i+1}. Found {len(most_similar_jd)} jd too similar.", "blue")
                print_colored(supervise_task, "blue")
                
                # Check if job descriptions are too similar
                if 'yes' in agent_2.answer(supervise_task).lower():
                    print_colored(f"Boss: Agent 1, these job descriptions are too similar. Do it again!", "red")

                break

if __name__ == "__main__":
    main()