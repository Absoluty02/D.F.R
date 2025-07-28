import csv
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd
from extractors import *
from openai import OpenAI
import anthropic

def new_entry_creator(index, gpt_result, gemini_result, claude_result):

    first_job_results = [
        index,
        gpt_result["mti"], gemini_result["mti"], claude_result["mti"],
        gpt_result["used_reasoning"], gemini_result["used_reasoning"], claude_result["used_reasoning"],
        gpt_result["explanation"], gemini_result["explanation"], claude_result["explanation"]
    ]

    second_job_results = [{} for i in range(11)]

    overall_result = {
        "gpt": gpt_result,
        "gemini": gemini_result,
        "claude": claude_result,
    }

    for llm_name, llm_result in overall_result.items():
        second_job_results[0][llm_name] = llm_result["second_job_csr"]["sentence"]
        second_job_results[1][llm_name] = llm_result["second_job_dr"]["sentence"]

        count = 2
        for key, value in llm_result["second_job_csr"]["inferences"].items():
            second_job_results[count][f"{llm_name}_csr"] = value
            count += 1

        count = 2
        for key, value in llm_result["second_job_dr"]["inferences"].items():
            second_job_results[count][f"{llm_name}_dr"] = value
            count += 1

    return first_job_results + second_job_results

def gemini_parser(gemini_client, model, user_prompt, system_prompt):
    response = gemini_client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        ),
        contents=user_prompt,
    )

    print(response.candidates[0].content.parts[0].text)

    result = manage_llm_response(response.candidates[0].content.parts[0].text)

    return result

def gpt_parser(gpt_client, model, user_prompt, system_prompt):

    response = gpt_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    print(response.choices[0].message.content)

    result = manage_llm_response(response.choices[0].message.content)

    return result

def claude_parser(claude_client, model, user_prompt, system_prompt):

    response = claude_client.messages.create(
        max_tokens=500,
        model=model,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    print(response.content[0].text)

    result = manage_llm_response(response.content[0].text)

    return result


def client():
    load_dotenv()

    gemini_client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
    gpt_client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
    claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_KEY"))

    df = pd.read_csv("v4_atomic_all.csv")
    sample = df.sample(n=27)

    ids = set()

    with open("responses.csv", mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row['id'])

    with open('prompt_template', 'r', encoding='utf-8') as f_in:
        instructions = f_in.read()

    for idx, row in sample.iterrows():
        if idx in ids:
            continue
        print(idx)
        json_content = row.to_json(orient="records")

        gemini_result = gemini_parser(gemini_client=gemini_client,model="gemini-2.5-flash",user_prompt=json_content,system_prompt=instructions)

        gpt_result = gpt_parser(gpt_client=gpt_client, model="gpt-4o-mini", user_prompt=json_content, system_prompt=instructions)

        claude_result = claude_parser(claude_client=claude_client, model="claude-sonnet-4-20250514", user_prompt=json_content, system_prompt=instructions)

        entry_row = new_entry_creator(index=idx, gpt_result=gpt_result, gemini_result=gemini_result, claude_result=claude_result)

        print(entry_row)

        with open("responses.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(entry_row)

if __name__ == "__main__":
    client()


