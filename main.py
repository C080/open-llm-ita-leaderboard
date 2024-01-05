import os


from model import llm_generate, generate_father_prompt, load_model, prep_list_of_prompts
from read_esco import read_dataset



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



if __name__ == "__main__":
    main()
