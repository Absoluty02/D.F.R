import re

def manage_llm_response(response):

    first_job_block, second_job_block = split_jobs_from_response(response)

    inference, why, used, not_used, explanation = extract_first_job_content(first_job_block)

    csr_dict, dr_dict = extract_second_job_content(second_job_block)

    reasoning = ""
    if "common sense" in used.lower():
        reasoning = "CSR"
    elif "default reasoning" in used.lower():
        reasoning = "DR"
    else:
        return ""

    result = {
        "mti": inference,
        "mti_why": why,
        "used_reasoning": reasoning,
        "not_used_reasoning": not_used,
        "explanation": explanation,
        "second_job_csr": csr_dict,
        "second_job_dr": dr_dict,
    }

    return result

def split_jobs_from_response(response_text):

    first_match = re.search(r'First Job\s*(.*?)\s*Second Job', response_text, re.DOTALL)
    first_block = first_match.group(1).strip() if first_match else ""

    second_match = re.search(r'Second Job\s*(.*)', response_text, re.DOTALL)
    second_block = second_match.group(1).strip() if second_match else ""

    return first_block, second_block


def extract_first_job_content(first_job_block) -> (str, str, str, str):

    inf_match = re.search(
        r"Most typical inference:\s*(.*?)\s*Why:\s*(.*?)(?:\s*Used reasoning:|$)",
        first_job_block,
        re.DOTALL
    )

    if inf_match:
        most_typical_inference = inf_match.group(1).strip()
        why_reasoning = inf_match.group(2).strip()
    else:
        most_typical_inference = ""
        why_reasoning = ""

    reasoning_match = re.search(
        r"Used reasoning:\s*(.*?)\s*Not used reasoning:\s*(.*)",
        first_job_block,
        re.DOTALL
    )

    if reasoning_match:
        used_reasoning = reasoning_match.group(1).strip()
        not_used_reasoning = reasoning_match.group(2).strip()
    else:
        used_reasoning = ""
        not_used_reasoning = ""

    explanation_match = re.search(
        r"Explanation:\s*(.*?)$",
        first_job_block,
        re.DOTALL
    )

    if explanation_match:
        explanation = explanation_match.group(1).strip()
    else:
        explanation = ""

    return most_typical_inference, why_reasoning, used_reasoning, not_used_reasoning, explanation


def extract_second_job_content(testo):

    parts = re.split(r'Default reasoning', testo, flags=re.IGNORECASE)

    common_text = parts[0].replace('Common sense', '').strip()
    common_sense = extract_inferences(common_text)

    default_text = parts[1].strip() if len(parts) > 1 else ""
    default_reasoning = extract_inferences(default_text)

    return common_sense, default_reasoning


def extract_inferences(testo):

    sentence = re.search(r'Sentence:\s*(.+)', testo).group(1).strip()

    inferences = {}
    inferences_section = re.search(r'Inferences:\s*(.+?)(?=Reasoning:|$)', testo, re.DOTALL)

    if inferences_section:
        inferences_text = inferences_section.group(1)
        matches = re.findall(r'^\s*([^:\n]+):\s*(.+)', inferences_text, re.MULTILINE)

        for key, value in matches:
            inferences[key.strip()] = value.strip()

    return {
        'sentence': sentence,
        'inferences': inferences
    }