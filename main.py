import os
from model import load_model, Agent
from read_esco import read_dataset
from utils import print_colored

def main():
    max_job_per_occupation = 3
    path = os.path.dirname(os.path.abspath(__file__))
    occupations = read_dataset(path)
    model, tokenizer = load_model()

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
        for j in range(max_job_per_occupation):
            answer = agent_1.answer(generate_task)
            print_colored(f"\nAgent 1: task #{i+1}-({j}/{max_job_per_occupation})", "green")
            print_colored(answer, "green")

            if j > 0:
                most_similar_jd = agent_1.remember(answer)
                # Construct task string
                supervise_task = f"Read the following job description: {most_similar_jd}.\
                \nNow read this second job description {answer}.\
                \nThey refer to the same job, but are these two job descriptions written almost the same? Answer Yes or no."
                print_colored(f"Agent 2: supervising task #{i+1}-({j}/{max_job_per_occupation}):", "blue")
                print_colored(supervise_task, "blue")
                
                # Check if job descriptions are too similar
                if 'yes' in agent_2.answer(supervise_task).lower():
                    print_colored(f"Boss: Agent 1, these job descriptions are too similar. Do it again!", "red")
                    j -= 1

        break

if __name__ == "__main__":
    main()