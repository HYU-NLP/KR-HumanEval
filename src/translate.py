from utils.utils import read_problems, write_jsonl
import deepl
import os


def translate(prompt, translator):
    # extract docstring
    doc_start = prompt.find('"""')
    doc_end = prompt.find('"""', doc_start + 3)
    
    if doc_start == -1 or doc_end == -1:
        doc_start = prompt.find("'''")
        doc_end = prompt.find("'''", doc_start + 3)
    
    docstring = prompt[doc_start + 3: doc_end]
    
    # translate
    translated_docstring = translator.translate_text(docstring, target_lang="KO").text
    new_prompt = prompt[:doc_start + 3] + translated_docstring + prompt[doc_end:]

    # print('original prompt:\n')
    # print(prompt)
    
    # print('new_prompt:\n')
    # print(new_prompt)
    
    return new_prompt

    
api_key = None # assign!
translator = deepl.Translator(api_key)
dataset = read_problems('HumanEval')

for pid in dataset:
    result = []
    example = dataset[pid]
    prompt = example['prompt']
    translated_prompt = translate(prompt, translator)
    example['translated_prompt'] = translated_prompt
    result.append(example)
    
    write_jsonl(save_path, result, append=True)