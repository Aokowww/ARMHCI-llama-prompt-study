# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 16:33:03 2025

@author: lzy98
"""


import pandas as pd
import requests
import time
from itertools import product
import re

# Configuration: API and model name (change as needed)
API_URL = "http://127.0.0.1:1234/v1/completions"
MODEL_NAME = "llama-3.2-3b-instruct"

# Load data
emotion_prompts = pd.read_csv("sentiment.csv")
emotion_shots = pd.read_csv("sentiment_shot.csv")

# Prompt formats and number of shots
formats = ['text', 'markdown', 'yaml', 'json']
shot_levels = [0, 1, 3, 5]

# Format few-shot examples (structured sentiment classification format)
def format_examples(examples, fmt):
    formatted = ""
    for ex in examples:
        input_text = ex['input'].strip()
        output_text = ex['output'].strip()
        if fmt == 'text':
            formatted += f"{input_text}\n{output_text}\n\n"
        elif fmt == 'json':
            formatted += f'{{"input": "{input_text}", "output": "{output_text}"}}\n'
        elif fmt == 'markdown':
            formatted += f"**{input_text}**\n**{output_text}**\n\n"
        elif fmt == 'yaml':
            formatted += f"- input: {input_text}\n  output: {output_text}\n"
    return formatted.strip()

# Format main prompt (instruction + actual sentence) to match examples
def format_main_prompt(main_prompt, fmt):
    instruction =  "Classify the sentiment of the following sentence(<output> is one of: positive, negative, or neutral,Respond strictly in the format: Label: <output> .DO NOT INCLUDE ANY EXPLANATION, TAGS, FORMATTING, OR EXTRA WORDS/EXAMPLES): \n"
    main_prompt = main_prompt.strip()

    if fmt == 'text':
        return f"{instruction}\n{main_prompt}"
    elif fmt == 'json':
        return f'{{"instruction": "{instruction}", "input": "{main_prompt}"}}'
    elif fmt == 'markdown':
        return f"**Instruction**: {instruction}\n**Input**: {main_prompt}"
    elif fmt == 'yaml':
        return f"instruction: {instruction}\ninput: {main_prompt}"
    else:
        return f"{instruction}\n{main_prompt}"

# More robust sentiment extractor (handles variations)
def extract_sentiment(output):
    lowered = output.lower()
    matches = re.findall(r'\b(positiv\w*|negativ\w*|neutral)\b', lowered)
    normalized = []
    for word in matches:
        if "positiv" in word:
            normalized.append("positive")
        elif "negativ" in word:
            normalized.append("negative")
        elif "neutral" in word:
            normalized.append("neutral")
    return normalized[0] if normalized else "unknown"

# LLM API call
def get_completion(prompt, max_tokens=64): # max_tokens can be adjusted depending on the model
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stop": None
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        if "choices" in result:
            return result["choices"][0]["text"].strip()
        else:
            return f"Error: {result}"
    except Exception as e:
        return f"Exception: {str(e)}"

# Main processing logic
def process_task(prompt_df, shot_df, save_name):
    results = []
    total = len(prompt_df)

    for i, row in prompt_df.iterrows():
        prompt_id = row['id']
        main_prompt = row['prompt'].strip()

        for shot_num, fmt in product(shot_levels, formats):
            if shot_num == 0:
                full_prompt = format_main_prompt(main_prompt, fmt)
                shots_used = ""
            else:
                examples = shot_df.iloc[:shot_num].to_dict(orient='records')
                shot_text = format_examples(examples, fmt)
                main_part = format_main_prompt(main_prompt, fmt)
                full_prompt = f"{shot_text}\n\n{main_part}"
                shots_used = shot_text

            print(f"[{i+1}/{total}] prompt_id={prompt_id} | shot={shot_num} | format={fmt}")
            model_output = get_completion(full_prompt)
            sentiment = extract_sentiment(model_output)

            results.append({
                "prompt_id": prompt_id,
                "prompt": main_prompt,
                "shot": shot_num,
                "format": fmt,
                "gpt_output": sentiment,
                "raw_output": model_output
            })

               # Save in real-time
            pd.DataFrame(results).to_csv(save_name, index=False)
            print(f" Saved result {len(results)} / {total * len(shot_levels) * len(formats)}\n")
            time.sleep(0.5)

    print(" All prompts processed.")
    return pd.DataFrame(results)

# Main execution
if __name__ == "__main__":
    process_task(emotion_prompts, emotion_shots, "sentiment_results.csv")


