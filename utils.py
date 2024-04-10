import csv
import re

def save_details_to_csv(details_result, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['hadm_id', 'precision', 'recall', 'f1', 'p_at_5', 'p_at_8', 'p_at_15', 'r_at_5', 'r_at_8', 'r_at_15', 'predicted_codes', 'ground_truth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for hadm_id in details_result:
            row = details_result[hadm_id]
            row['hadm_id'] = hadm_id
            writer.writerow(row)

def get_content_between_a_b(a,b,text):
    match = re.search(f"{a}(.*?)\n{b}", text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return ""

def split_into_subsections(text, max_length=250):
    words = text.split()
    subsections = []
    current_section = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_section.append(word)
            current_length += len(word) + 1
        else:
            subsections.append(' '.join(current_section))
            current_section = [word]
            current_length = len(word) + 1
    if current_section:
        subsections.append(' '.join(current_section))
    return subsections