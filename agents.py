from openai import OpenAI
import os
import re
from codes import codes_50

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", 'your_api_key_here'))


def icd_subsection_base(sub_section_text, previous_response):
    command_template = \
    """
    You are reading a sub-section of a patient record and the report of last ICD-9 coder. Read the following patient report carefully. Summarize the main points of the section, highlighting key medical conditions, treatments, and procedures mentioned. Then, start the coding process by assigning the appropriate ICD-9 codes to each condition you identify. For each code you assign, provide the direct evidence from the patient's report that supports the use of that particular code. Ensure your coding is precise and justified by the patient's documented symptoms, diagnoses, and procedures.

    You will complete the following writing tasks：
    1. Output Short-Term Summary: Summarize the main points of the given sub-section to help next coder understand what this sub-section describes.
    2. Review and Assign ICD-9 Codes: Assign the appropriate ICD-9 codes to each condition you identify in the patient report and give the direct evidence. In addition to generating code that you think must be correct and undecidable code, you also need to check the previous coder for undecidable code. If there is new information in your sub-section to confirm or exclude this code, please update this code to the correct code area or delete it from the undecided code area
    3. Leave comments and instructions for next coder: Leave comment and instruction for the next coder and let them know if you have any code that needs further confirmation.
    4. Update long-term summary: Update the long-term summary and notes of this report to facilitate the summary and noteworthy of this section after all sub-sections have been completed.This is an iterative update process, the text length should not be much longer each time, you need to constantly summarize the previous long summary and your short summary to form the new long summary, while deleting any information that you think the next coder will not need to know.
    5. Correct Codes are added or modified on the basis of previous ones, which is a cumulative process in iteration

    Attention:
    !!!Keep in mind that you do not assign an ICD-9 code to every disease that is present, but to the disease that the patient is currently having. Just because someone has a disease in the past doesn't mean they have it now!!!
    !!!Please assign code very carefully and conservatively, usually a sub-section will only get a new code or can not allocate the relevant code, unless you are absolutely sure that the sub-section will get multiple 100 percent correct code.!!!
    !!!Please note that it is possible for the previous coder to assign the ICD-10 code or the wrong name to the code he assigned or duplicate the output. If this happens, please correct the wrong code (and corresponding name) and delete the duplicate code.!!!
    !!!Please note that the previous coder may have assigned the correct code, but the evidence he provided is not sufficient. If you think the code is correct, but the evidence is not sufficient, please provide additional evidence to support the code.!!!

    ***sub-section Begins***
    {sub_section_text}
    ***sub-section Ends***

    ***Report of last ICD-9 coder Begins***
    Last ICD-9 coder's report: (If it is empty, then you are assigned a sub-section at the beginning of the section, please ignore this part. Your short and long term summary will be the same for your first sub-section)
    {previous_response},
    ***Report of last ICD-9 coder Ends***

    Follow the template below to output the result, and replace the placeholder with your response:
    ***Template Begins***

    ##Short-Term Summary of this sub-section##
    [Insert the summary of this sub-section here, highlighting key points and medical information.], around 2-3 sentences.

    ##cumulative Correct Codes for this report##
    [Update the correct codes for this report here according to previous correct codes and your new correct codes.]
    [Correct Codes are added or modified on the basis of previous codes, which is a cumulative process in iteration. Please dynamically update Correct Codes during iteration]
    Previous coder's correct codes: 
    "Code1", [Accurate Name of Code1], ["Original text from sub-section containing direct evidence1 supporting Code1"], [Reason for using evidence1 to assign Code1]
    "Code2", [Accurate Name of Code2], ["Original text from sub-section containing direct evidence2 supporting Code2"], [Reason for using evidence2 to assign Code2]
    ...
    Additional correct codes in this sub-section:
    [If you have new correct codes in this sub-section, please add them here.]
    "Code1", [Accurate Name of Code1], ["Original text from sub-section containing direct evidence2 supporting Code"], [Reason for using evidence2 to assign Code1]
    ...

    ##Undecidable Codes##
    "Code1", [Accurate Name of Code1], ["Relevant text from sub-section to support Code1"], [Why do you think this code is undecidable? What other information do you need to confirm.]
    "Code2", [Accurate Name of Code2], ["Relevant text from sub-section to support Code2"], [Why do you think this code is undecidable? What other information do you need to confirm.]
    ...

    ##Comments and Instructions##
    [Leave comments and instructions for the next coder here, if any.]

    ##Long-Term Summary of this report##
    [Update the long-term summary and notes of the patient report here according to previous long-term summary and your short-term summary of this sub-section, highlighting key points and medical information.]

    ***Template Ends***

    """
    try:
        # Generate completion from OpenAI's chat model
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            timeout=100,
            messages=[
                {"role": "user", "content": "You are a professional ICD-9 coder with extensive experience in the medical coding field. You have a deep understanding of the ICD-9 coding system and are familiar with a wide range of medical terminologies and procedures. Your role involves reading and interpreting detailed patient reports, identifying relevant medical conditions, and assigning accurate ICD-9 codes based on the information provided. You pay close attention to details and ensure that each code is supported by direct evidence from the patient's report."},
                {"role": "user", "content": "You only need to consider ICD-9 codes."},
                #{"role": "system", "content": f"You only need to consider the following 50 common icd-9 codes from codes_50: {codes_50}"},
                {"role": "user", "content": command_template.format(sub_section_text=sub_section_text, previous_response=previous_response)},
            ]
        )
        return completion.choices[0].message.content  # Return the predicted result
    except Exception as e:
        return "Error: " + str(e)
    
def icd_subsection_board(sub_section_text, previous_response):
    introduction = \
    """
    Here is the introduction for each part in template.
    1. Short-Term Summary of this sub-section: Summarize the main points of the given sub-section to help next coder understand what this sub-section describes.
    2. Correct Codes for this sub-section: This section is the code you assign to this sub-section that you think is correct, but then you need to do some strict sorting to add the code to the history correct codes board or undecidable codes or choose to ignore or delete the code. 
    3. Undecidable Codes: This section is the code you assign to this sub-section that you think is undecidable, but then you need to do some strict sorting to add the code to the history correct codes board or delete the code.
    4. History Correct Codes Board: The code in this section is from the previous iteration, which is the codes assigned by the previous coder to the previous sub-sections. If you think your code is correct, you can put the code you think must be correct into this section.
    5. Comments and Instructions: This section is the comments and instructions you leave for the next coder, which is the information you think the next coder needs to know and the code that needs to be confirmed.
    6. Long-Term Summary of this report: This section is the long-term summary and notes of the patient report, which is the summary and noteworthy of this section after all sub-sections have been completed. This is an iterative update process, the text length should not be much longer each time, you need to constantly summarize the previous long summary and your short summary to form the new long summary, while deleting any information that you think the next coder will not need to know.
    """
    command_template = \
    """

    You are reading a sub-section of a patient record and the report of last ICD-9 coder. Read the following patient report carefully. Summarize the main points of the section, highlighting key medical conditions, treatments, and procedures mentioned. Then, start the coding process by assigning the appropriate ICD-9 codes to each condition you identify. For each code you assign, provide the direct evidence from the patient's report that supports the use of that particular code. Ensure your coding is precise and justified by the patient's documented symptoms, diagnoses, and procedures.
    
    You will complete the following writing tasks：
    1. Output Short-Term Summary: Summarize the main points of the given sub-section to help next coder understand what this sub-section describes.
    2. Review and Assign ICD-9 Codes: Assign the appropriate ICD-9 codes to each condition you identify in the patient report and give the direct evidence. In addition to generating code that you think must be correct and undecidable code, you also need to check the previous coder for undecidable code. If there is new information in your sub-section to confirm or exclude this code, please update this code to the correct code history or delete it from the undecided code area.
    3. Leave comments and instructions for next coder: Leave comment and instruction for the next coder and let them know if you have any code that needs further confirmation.
    4. Update long-term summary: Update the long-term summary and notes of this report to facilitate the summary and noteworthy of this section after all sub-sections have been completed.This is an iterative update process, the text length should not be much longer each time, you need to constantly summarize the previous long summary and your short summary to form the new long summary, while deleting any information that you think the next coder will not need to know.
    
    Attention:
    !!!Keep in mind that you do not assign an ICD-9 code to every disease that is present, but to the disease that the patient is currently having. Just because someone has a disease in the past doesn't mean they have it now!!!
    !!!Please assign code very carefully and conservatively, usually a sub-section will only get a new code or can not allocate the relevant code, unless you are absolutely sure that the sub-section will get multiple 100 percent correct code.!!!
    !!!Please note that it is possible for the previous coder to assign the ICD-10 code or the wrong name to the code he assigned or duplicate the output. If this happens, please correct the wrong code (and corresponding name) and delete the duplicate code.!!!
    !!!Please note that the previous coder may have assigned the correct code, but the evidence he provided is not sufficient. If you think the code is correct, but the evidence is not sufficient, please provide additional evidence to support the code.!!!

    ***sub-section Begins***
    {sub_section_text}
    ***sub-section Ends***

    ***Report of last ICD-9 coder Begins***
    Last ICD-9 coder's report: (If it is empty, then you are assigned a sub-section at the beginning of the section, please ignore this part. Your short and long term summary will be the same for your first sub-section)
    {previous_response}, 
    ***Report of last ICD-9 coder Ends***

    Follow the template below to output the result:
    ***Template Begins***

    ##Short-Term Summary of this sub-section##
    [Insert the summary of this sub-section here, highlighting key points and medical information.], around 2-3 sentences.

    ##Correct Codes for this sub-section##
    If you are absolutely sure that the inferred code in this sub-section is Correct, add the code to the History Correct Codes Board. If you are not sure, please add it to the Undecidable Codes Board or delete it.
    "Code1", [Accurate Name of Code1], ["Original text from sub-section containing direct evidence1 supporting Code1"], [Reason for using evidence1 to assign Code1]
    "Code2", [Accurate Name of Code2], ["Original text from sub-section containing direct evidence2 supporting Code2"], [Reason for using evidence2 to assign Code2]
    ...

    ##Undecidable Codes Board##
    Attention: Please check the previous coder's undecidable code and provide new information to confirm or exclude this code. Please only move the code to the hisory correct codes board if you are absolutely sure that the code is correct.
    "Code1", [Accurate Name of Code1], ["Relevant text from sub-section to support Code1"], [Why do you think this code is undecidable? What other information do you need to confirm.]
    "Code2", [Accurate Name of Code2], ["Relevant text from sub-section to support Code2"], [Why do you think this code is undecidable? What other information do you need to confirm.]
    ...

    ##History Correct Codes Board##
    Attention: Please only add new codes or delete ABSOLUTELY wrong codes from the history correct code board. Do not modify the existing codes unless you are absolutely sure that the code is wrong.
    "Code1", [Accurate Name of Code1], ["Original text from sub-section containing direct evidence1 supporting Code1"], [Reason for using evidence1 to assign Code1]
    "Code2", [Accurate Name of Code2], ["Original text from sub-section containing direct evidence2 supporting Code2"], [Reason for using evidence2 to assign Code2]
    ...

    ##Comments and Instructions##
    [Leave comments and instructions for the next coder here. If you have any code that needs further confirmation, please mention it here.]

    ##Long-Term Summary of this report##
    [Update the long-term summary and notes of the patient report here according to previous long-term summary and your short-term summary of this sub-section, highlighting key points and medical information.]
    
    ***Template Ends***

    """
    try:
        # Generate completion from OpenAI's chat model
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            timeout=100,
            messages=[
                {"role": "user", "content": "You are a professional ICD-9 coder with extensive experience in the medical coding field. You have a deep understanding of the ICD-9 coding system and are familiar with a wide range of medical terminologies and procedures. Your role involves reading and interpreting detailed patient reports, identifying relevant medical conditions, and assigning accurate ICD-9 codes based on the information provided. You pay close attention to details and ensure that each code is supported by direct evidence from the patient's report."},
                {"role": "user", "content": "You only need to consider ICD-9 codes. You do not need to consider ICD-10 codes or any other coding systems. Your primary goal is to assign the most accurate ICD-9 codes to each condition identified in the patient report, ensuring that the codes are justified by the documented symptoms, diagnoses, and procedures."},
                {"role": "system", "content": introduction},
                #{"role": "system", "content": f"You only need to consider the following 50 common icd-9 codes from codes_50: {codes_50}"},
                {"role": "user", "content": command_template.format(sub_section_text=sub_section_text, previous_response=previous_response)},
            ]
        )
        return completion.choices[0].message.content  # Return the predicted result
    except Exception as e:
        return "Error: " + str(e)

# function for merge_section in second round
def merge_section(sections):
    command_template = \
    """
    Previous coders has assigned the following ICD-9 codes to each section of the patient report. Please review the summary of each section, codes and the direct evidence from the patient's report that supports the use of each code.
    Then, assign the final ICD-9 codes to the patient report based on the information provided. Ensure your coding is precise and justified by the patient's documented symptoms, diagnoses, and procedures.

    ***Previous Coders' Assignments***
    {sections_text}
    ***Previous Coders' Assignments Ends***

    Follow the template below to output the result:

    ***Template Begins***
    ##Summary of the whole report##
    [Insert the summary of the patient report here, highlighting key points and medical information.]

    ##Codes ranked by level of confidence##
    "Code1", Reason for choosing Code1, [Confidence level (1-10): 10 is the most confident]
    "Code2", Reason for choosing Code2, [Confidence level (1-10): 10 is the most confident]
    "Code3", Reason for choosing Code3, [Confidence level (1-10): 10 is the most confident]
    ...
    ***Template Ends***

    """

    try:

        # Generate completion from OpenAI's chat model
        completion = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            temperature=0.0,
            timeout=100,
            messages=[
                {"role": "system", "content": "You are a professional ICD-9 auditor employed by an insurance company with extensive experience in the field of medical coding and you want to be able to obtain the most accurate ICD-9 codes to reduce costs for the insurance company. For a patient applying for a claim, the patient hired different ICD-9 coders to be in charge of different sections to help him generate as many ICD-9 codes as possible to obtain more claims."},
                {"role": "system", "content": "Your task is to check and eliminate the wrong ICD-9 codes according to the assignments of ICD-9 coders, leaving the most accurate ICD codes. Based on the summary of the corresponding sections recorded by these ICD coders, you should refer to the original text containing the direct evidence and the reason for obtaining the corresponding ICD-9 code, so as to exclude those inaccurate codes to obtain the final code. You should take a hard look at the codes that coders get, thereby reducing the cost to your insurance company."},
                #{"role": "system", "content": f"You only need to consider the following 50 common icd-9 codes from codes_50: {codes_50}"},
                {"role": "user", "content": command_template.format(sections_text=sections)},
            ]
        )
        return completion.choices[0].message.content  # Return the predicted result
    except Exception as e:
        return "Error: " + str(e)

# Original function for section_base
def section_base(section_name, section_text):
    command_template = \
    """
    Read the following patient report section carefully. Summarize the main points of the section, highlighting key medical conditions, treatments, and procedures mentioned. Then, start the coding process by assigning the appropriate ICD-9 codes to each condition you identify. For each code you assign, provide the direct evidence from the patient's report that supports the use of that particular code. Ensure your coding is precise and justified by the patient's documented symptoms, diagnoses, and procedures.
    Prioritize your final output in order of confidence and avoid generating duplicate codes.

    ***Report Begins***
    {section_name}:
    {section_text}
    ***Report Ends***

    Follow the template below to output the result:
    ***Template Begins***
    ##Summary of the section: {section_name}##
    [Insert the summary of the patient report here, highlighting key points and medical information.]

    ##Codes ranked by level of confidence##
    "Code1", [Accurate Name of Code1], ["Original text from patient reports containing direct evidence1 supporting Code1"], [Reason for using evidence1 to assign Code1]
    "Code2", [Accurate Name of Code2], ["Original text from patient reports containing direct evidence2 supporting Code2"], [Reason for using evidence2 to assign Code2]
    ...
    ***Template Ends***

    """
    #print(command_template.format(whole_text=whole_text))
    try:
        # Generate completion from OpenAI's chat model
        completion = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            temperature=0.0,
            timeout=100,
            messages=[
                {"role": "user", "content": "You are a professional ICD-9 coder with extensive experience in the medical coding field. You have a deep understanding of the ICD-9 coding system and are familiar with a wide range of medical terminologies and procedures. Your role involves reading and interpreting detailed patient reports, identifying relevant medical conditions, and assigning accurate ICD-9 codes based on the information provided. You pay close attention to details and ensure that each code is supported by direct evidence from the patient's report."},
                {"role": "user", "content": "You only need to consider icd-9 codes."},
                #{"role": "system", "content": f"You only need to consider the following 50 common icd-9 codes from codes_50: {codes_50}"},
                {"role": "user", "content": command_template.format(section_name=section_name, section_text=section_text)},
            ]
        )
        return completion.choices[0].message.content  # Return the predicted result
    except Exception as e:
        return "Error: " + str(e)

