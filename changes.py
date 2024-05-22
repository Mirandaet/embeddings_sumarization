import subprocess
from openai import OpenAI
import os

def suggest_code_change(file_path, description):
    # Construct the prompt for the LLM
    prompt = f"Modify the following Python code according to the user's request: {description}\n\n"

    # Read the current code
    with open(file_path, 'r') as file:
        code_content = file.read()

    # Append the existing code to the prompt
    prompt += code_content

    # Query the LLM for a modification
    response = query_gpt4(prompt)

    # Output the suggested change
    return response

def apply_change(file_path, new_code):
    # Write the new code to the file
    with open(file_path, 'w') as file:
        file.write(new_code)
    
    # Use git to commit changes
    subprocess.run(["git", "add", file_path], check=True)
    subprocess.run(["git", "commit", "-m", "Applied code modification"], check=True)
    subprocess.run(["git", "push"], check=True)

def review_changes(file_path, suggested_changes):
    with open(file_path, 'r') as file:
        code_content = file.read()
    
    print("\n\nCURRENT FILE: \n\n")
    print(code_content)
    print("\n\n\nSUGGESTED CHANGES: \n\n")
    print(suggested_changes)
    confirm = input("Do you want to apply these changes? (yes/no): ")
    if confirm.lower() == 'yes':
        apply_change(file_path, suggested_changes)
        print("Changes applied and committed.")
    else:
        print("Changes rejected.")

def query_gpt4(prompt):
        client = OpenAI(
            # this is also the default, it can be omitted
            api_key=os.environ['OPENAI_API_KEY'],
        )
        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You suggest code improvements according to the users description, you answer ONLY in code, no other characters but pure python code, DO NOT ADD ``` AT THE BEGGIN OR END OF THE FILE"},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content