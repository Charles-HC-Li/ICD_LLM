import json
import threading
import random
import time
import os
from utils import save_details_to_csv
from evaluation import print_result
from process import process_case

lock = threading.Lock()

def main():
    print("Current Working Directory:", os.getcwd())
    # Load dataset from JSON file
    with open(r"D:\baseline\code\data\dataset_30.json", 'r') as file:
        data = json.load(file)

    data = data[:1]

    overall_results = {'precision': [], 'recall': [], 'f1': []}
    summary_results = {'precision': [], 'recall': [], 'f1': []}
    details_result = {}
    details_result_summary = {}

    # Create threads
    threads = []
    for case in data:
        thread = threading.Thread(target=process_case, args=(case, overall_results, details_result, summary_results, details_result_summary,lock))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Print results
    print_result(overall_results, summary_results, details_result, details_result_summary)

    # Save details_result to CSV
    save_details_to_csv(details_result, 'GPT4_new_merged_full.csv')
    save_details_to_csv(details_result_summary, 'GPT4_summary_merged_full.csv')

if __name__ == "__main__":
    main()
