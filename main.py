import os


from model import llm_generate, generate_father_prompt, load_model, prep_list_of_prompts, Agent
from read_esco import read_dataset
from utils import print_colored



def main():
    print("Starting...")
    path = os.path.dirname(os.path.abspath(__file__))
    # import dataset dataset
    occupations = read_dataset(path)
    # load model
    model, tokenizer = load_model()
    # prepare the first prompt
    occupations_dict = {}
    for _, row in occupations.iterrows():
        father_prompt = generate_father_prompt(row)
        job_descriptions_prompt = llm_generate(model, tokenizer, father_prompt, 'cuda') #generate the first prompt
        job_descriptions_prompt_list =      (job_descriptions_prompt)
        prompt_oja_dict = {}
        for i, job_description_prompt in enumerate(job_descriptions_prompt_list):
            print(f"Generating Prompt {i+1}")
            completion = llm_generate(model, tokenizer, job_description_prompt, 'cuda') #generate the job description
            prompt_oja_dict[job_description_prompt] = completion
            print(f"Prompt {i+1} completed")
            print(prompt_oja_dict)
            break
        occupations_dict[row['preferredLabel']] = prompt_oja_dict
        break
    #print(occupations_dict)


def main_2():
    path = os.path.dirname(os.path.abspath(__file__))
    occupations = read_dataset(path)
    model, tokenizer = load_model()
    for i, row in occupations.iterrows():
        for j in range(3):
            print_colored(f"Boss: incoming task #{i+1} for Agent 1", "red")
            task = f"Generate a realistic job description for the role of {row['preferredLabel']},\
                    \nwe are talking about a {row['description']}. It is also know as:\
                    \n{row['altLabels']}."
            print_colored(f"Agent 1: receiving task #{i+1}:", "blue")
            print_colored(task, "blue")
            worker_1 = Agent(model, tokenizer, 'cuda')
            result = worker_1.answer(task)
            print_colored(f"Agent 1: task #{i+1} completed", "green")
            print_colored(result, "green")
            if j == 0:
                print_colored(f"Boss: incoming task #{i+1}-{j+1} for Agent 2", "red")
                last_job_description = worker_1.remember()
                task = f"Read the following job description: {last_job_description}.\
                    \nIs too much similar to this one?: {result}.\
                    \nYes or No?"
                print_colored(f"Agent 2: receiving task #{i+1}-{j+1}:", "blue")
                print_colored(task, "blue")
                worker_2 = Agent(model, tokenizer, 'cuda')
                result = worker_2.answer(task)
                break







if __name__ == "__main__":
    main_2()
