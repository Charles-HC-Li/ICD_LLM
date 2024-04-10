import re
import time
from collections import OrderedDict
from codes import codes_50
from evaluation import evaluate_case
from utils import split_into_subsections, get_content_between_a_b
from agents import icd_subsection_base, icd_subsection_board, merge_section

def process_case(case, overall_results, details_result, summary_results, details_result_summary, lock):
    hadm_id = case['hadm_id']
    sections = case['sections']
    result = {"hadm_id": hadm_id, "sections": {}}
    ground_truth = case['LABELS']
    ground_truth.sort()
    codes = []
    section_responses = []

    # First round: Process each subsection of each section
    for section, text in sections.items():
        subsections = split_into_subsections(text)
        previous_subsection_response = "This is the beginning of the cycle and there is no previoes response!"
        for subsection in subsections:
            flag = False
            while not flag:
                # Pass the previous subsection response along with the current subsection
                gpt_response = icd_subsection_base(subsection, previous_subsection_response)
                print(gpt_response)
                if 'Error' in gpt_response:
                    print(gpt_response)
                    time.sleep(5)
                    continue
                previous_subsection_response = gpt_response
                flag = True
        # Second round: Combine the responses and call GPT API for section summary       
        #section_response = call_my_gpt_api_2(previous_subsection_response)
        section_response = previous_subsection_response
        result["sections"][section] = section_response
        section_responses.append(section_response)
        temp_content = get_content_between_a_b('##History', '##Comments', section_response)
        codes_sec = re.findall(r'[E|V]?\d{2,3}\.\d{1,2}', temp_content)
        codes.extend(codes_sec)

    # Third round: Combine section responses and call GPT API for case summary
    combined_response = "\n".join(section_responses)
    case_response = merge_section(combined_response)
    print(case_response)
    result["summary"] = case_response
    summary_codes = re.findall(r'[E|V]?\d{2,3}\.\d{1,2}', case_response)
    codes.extend(summary_codes)

    codes = list(OrderedDict.fromkeys(codes))
    summary_codes = list(OrderedDict.fromkeys(summary_codes))

    # Acquire the lock before modifying shared resources
    lock.acquire()
    try:
        ground_truth.sort()
        codes.sort()
        summary_codes.sort()
        evaluation_result, evaluation_result_summary = evaluate_case(codes, summary_codes, ground_truth)

        overall_results['precision'].append(evaluation_result['precision'])
        summary_results['precision'].append(evaluation_result_summary['precision'])
        overall_results['recall'].append(evaluation_result['recall'])
        summary_results['recall'].append(evaluation_result_summary['recall'])
        overall_results['f1'].append(evaluation_result['f1'])
        summary_results['f1'].append(evaluation_result_summary['f1'])

        # Store detailed results for micro evaluation
        details_result[hadm_id] = evaluation_result
        details_result_summary[hadm_id] = evaluation_result_summary
    finally:
        # Release the lock regardless of whether an exception occurred
        lock.release()