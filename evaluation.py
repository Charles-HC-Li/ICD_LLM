def calculate_metrics(codes, ground_truth):
    # Precision, recall, F1 score
    precision = len(set(codes).intersection(ground_truth)) / len(codes) if len(codes) > 0 else 1
    recall = len(set(codes).intersection(ground_truth)) / len(ground_truth) if len(ground_truth) > 0 else 1
    f1 = 2 * (precision * recall) / (precision + recall+1e-9)
    return precision, recall, f1

def calculate_p_at_k(codes, ground_truth, k):
    top_k_codes = codes[:k]
    relevant_codes = len(set(top_k_codes).intersection(ground_truth))
    return relevant_codes / k

def calculate_r_at_k(codes, ground_truth, k):
    top_k_codes = codes[:k]
    relevant_codes = len(set(top_k_codes).intersection(ground_truth))
    return relevant_codes / len(ground_truth)

def calculate_overall_p_at_k(details_result, k):
    relevant_codes = 0
    for hadm_id in details_result:
        top_k_codes = details_result[hadm_id]['predicted_codes'][:k]
        relevant_codes += len(set(top_k_codes).intersection(details_result[hadm_id]['ground_truth']))
    return relevant_codes / (k * len(details_result))

def calculate_overall_r_at_k(details_result, k):
    relevant_codes = 0
    total_ground_truth = 0
    for hadm_id in details_result:
        top_k_codes = details_result[hadm_id]['predicted_codes'][:k]
        relevant_codes += len(set(top_k_codes).intersection(details_result[hadm_id]['ground_truth']))
        total_ground_truth += len(details_result[hadm_id]['ground_truth'])
    return relevant_codes / total_ground_truth

def evaluate_case(codes, summary_codes, ground_truth):
    # Compare the codes with the ground truth
    precision, recall, f1 = calculate_metrics(codes, ground_truth)
    precision_summary, recall_summary, f1_summary = calculate_metrics(summary_codes, ground_truth)
    p_at_5 = calculate_p_at_k(codes, ground_truth, 5)
    p_at_5_summary = calculate_p_at_k(summary_codes, ground_truth, 5)
    p_at_8 = calculate_p_at_k(codes, ground_truth, 8)
    p_at_8_summary = calculate_p_at_k(summary_codes, ground_truth, 8)
    p_at_15 = calculate_p_at_k(codes, ground_truth, 15)
    p_at_15_summary = calculate_p_at_k(summary_codes, ground_truth, 15)
    r_at_5 = calculate_r_at_k(codes, ground_truth, 5)
    r_at_5_summary = calculate_r_at_k(summary_codes, ground_truth, 5)
    r_at_8 = calculate_r_at_k(codes, ground_truth, 8)
    r_at_8_summary = calculate_r_at_k(summary_codes, ground_truth, 8)
    r_at_15 = calculate_r_at_k(codes, ground_truth, 15)
    r_at_15_summary = calculate_r_at_k(summary_codes, ground_truth, 15)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'p_at_5': p_at_5,
        'p_at_8': p_at_8,
        'p_at_15': p_at_15,
        'r_at_5': r_at_5,
        'r_at_8': r_at_8,
        'r_at_15': r_at_15,
        'predicted_codes': codes,
        'ground_truth': ground_truth
    }, {
        'precision': precision_summary,
        'recall': recall_summary,
        'f1': f1_summary,
        'p_at_5': p_at_5_summary,
        'p_at_8': p_at_8_summary,
        'p_at_15': p_at_15_summary,
        'r_at_5': r_at_5_summary,
        'r_at_8': r_at_8_summary,
        'r_at_15': r_at_15_summary,
        'predicted_codes': summary_codes,
        'ground_truth': ground_truth
    }


def print_result(overall_results, summary_results, details_result, details_result_summary):
    # Calculate MACRO and MICRO metrics
    macro_precision = sum(overall_results['precision']) / len(overall_results['precision'])
    macro_precision_summary = sum(summary_results['precision']) / len(summary_results['precision'])
    macro_recall = sum(overall_results['recall']) / len(overall_results['recall'])
    macro_recall_summary = sum(summary_results['recall']) / len(summary_results['recall'])
    macro_f1 = sum(overall_results['f1']) / len(overall_results['f1'])
    macro_f1_summary = sum(summary_results['f1']) / len(summary_results['f1'])

    total_tp = sum([len(set(details_result[hadm_id]['predicted_codes']).intersection(details_result[hadm_id]['ground_truth'])) for hadm_id in details_result])
    total_tp_summary = sum([len(set(details_result_summary[hadm_id]['predicted_codes']).intersection(details_result_summary[hadm_id]['ground_truth'])) for hadm_id in details_result_summary])
    total_fp = sum([len(details_result[hadm_id]['predicted_codes']) - len(set(details_result[hadm_id]['predicted_codes']).intersection(details_result[hadm_id]['ground_truth'])) for hadm_id in details_result])
    total_fp_summary = sum([len(details_result_summary[hadm_id]['predicted_codes']) - len(set(details_result_summary[hadm_id]['predicted_codes']).intersection(details_result_summary[hadm_id]['ground_truth'])) for hadm_id in details_result_summary])
    total_fn = sum([len(details_result[hadm_id]['ground_truth']) - len(set(details_result[hadm_id]['predicted_codes']).intersection(details_result[hadm_id]['ground_truth'])) for hadm_id in details_result])
    total_fn_summary = sum([len(details_result_summary[hadm_id]['ground_truth']) - len(set(details_result_summary[hadm_id]['predicted_codes']).intersection(details_result_summary[hadm_id]['ground_truth'])) for hadm_id in details_result_summary])

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1
    micro_precision_summary = total_tp_summary / (total_tp_summary + total_fp_summary) if (total_tp_summary + total_fp_summary) > 0 else 1
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1
    micro_recall_summary = total_tp_summary / (total_tp_summary + total_fn_summary) if (total_tp_summary + total_fn_summary) > 0 else 1
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-9)
    micro_f1_summary = 2 * (micro_precision_summary * micro_recall_summary) / (micro_precision_summary + micro_recall_summary + 1e-9)

    print("Macro Evaluation")
    print(f"Macro precision: {macro_precision}")
    print(f"Macro precision summary: {macro_precision_summary}")
    print(f"Macro recall: {macro_recall}")
    print(f"Macro recall summary: {macro_recall_summary}")
    print(f"Macro F1: {macro_f1}")
    print(f"Macro F1 summary: {macro_f1_summary}")

    print("Micro Evaluation")
    print(f"Micro precision: {micro_precision}")
    print(f"Micro precision summary: {micro_precision_summary}")
    print(f"Micro recall: {micro_recall}")
    print(f"Micro recall summary: {micro_recall_summary}")
    print(f"Micro F1: {micro_f1}")
    print(f"Micro F1 summary: {micro_f1_summary}")

    # Calculate overall P@k and R@k
    overall_p_at_5 = calculate_overall_p_at_k(details_result, 5)
    overall_p_at_5_summary = calculate_overall_p_at_k(details_result_summary, 5)
    overall_p_at_8 = calculate_overall_p_at_k(details_result, 8)
    overall_p_at_8_summary = calculate_overall_p_at_k(details_result_summary, 8)
    overall_p_at_15 = calculate_overall_p_at_k(details_result, 15)
    overall_p_at_15_summary = calculate_overall_p_at_k(details_result_summary, 15)
    overall_r_at_5 = calculate_overall_r_at_k(details_result, 5)
    overall_r_at_5_summary = calculate_overall_r_at_k(details_result_summary, 5)
    overall_r_at_8 = calculate_overall_r_at_k(details_result, 8)
    overall_r_at_8_summary = calculate_overall_r_at_k(details_result_summary, 8)
    overall_r_at_15 = calculate_overall_r_at_k(details_result, 15)
    overall_r_at_15_summary = calculate_overall_r_at_k(details_result_summary, 15)

    print("Overall P@k and R@k")
    print(f"Overall P@5: {overall_p_at_5}, P@8: {overall_p_at_8}, P@15: {overall_p_at_15}, R@5: {overall_r_at_5}, R@8: {overall_r_at_8}, R@15: {overall_r_at_15}")
    print(f"Overall P@5 summary: {overall_p_at_5_summary}, P@8 summary: {overall_p_at_8_summary}, P@15 summary: {overall_p_at_15_summary}, R@5 summary: {overall_r_at_5_summary}, R@8 summary: {overall_r_at_8_summary}, R@15 summary: {overall_r_at_15_summary}")
