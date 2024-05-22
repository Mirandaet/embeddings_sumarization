import os
import json
from openai import OpenAI
import glob


class CodeSummarizer:
    def __init__(self, model, api_key):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def query_gpt4(self, file_path):
        client = OpenAI(
            # this is also the default, it can be omitted
            api_key=os.environ['OPENAI_API_KEY'],
        )
        with open(file_path, 'r') as file:
            code_content = file.read()
        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You summarize python code to a shorter description of the file"},
                {"role": "user", "content": code_content}
            ]
        )
        return completion.choices[0].message.content

    def summarize_code(self, file_path):
        with open(file_path, 'r') as file:
            code_content = file.read()
        response = self.client.completions.create(
            model=self.model,
            prompt=f"Summarize this Python code: \n\n{code_content}",
            max_tokens=150
        )
        return response.choices[0].text.strip()

    def process_directory(self, directory_path):
        summaries = {}
        for filename in glob.glob(os.path.join(directory_path, '*.py')):
            summary = self.query_gpt4(filename)
            summaries[filename] = summary
            print(filename, "done")
        return summaries

    def save_summaries(self, summaries, output_file='summaries.json'):
        with open(output_file, 'w') as f:
            json.dump(summaries, f, indent=4)


# Usage
if "__name__" == "__main__":
    api_key = os.environ['OPENAI_API_KEY']
    model = 'gpt-4-instruct'  # Or any other suitable model
    summarizer = CodeSummarizer(model, api_key)
    directory_path = './codebase/Uppgift_2'
    summaries = summarizer.process_directory(directory_path)
    summarizer.save_summaries(summaries)
