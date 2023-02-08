import os
import openai

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

openai.api_key = open_file('openaiapikey.txt')

while True:
    prompt = input("\n Ask OpenAI Anything: ")
    completions = openai.Completion.create(prompt = prompt,
                                            engine='text-ada-001', #engine="text-davinci-003",
                                            max_tokens=100)   # https://platform.openai.com/tokenizer , Get Started Introduction - Key concepts - Token, https://openai.com/api/pricing/
    #print(completions)  #이렇게 하면 전체 JSON 포맷으로 표시 된다. 그래서 아래처럼 해야 한다.
    completion = completions.choices[0].text
    print(completion)
                               