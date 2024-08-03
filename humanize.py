from openai import OpenAI
import json
import re

# Initialize the OpenAI client
client = OpenAI(
    api_key='')

instructions_model_name = "ft:gpt-4o-mini-2024-07-18:prosp::9rgTmhF3"
humanizer_model_name = "ft:gpt-4o-mini-2024-07-18:prosp::9rgUTEhx"


# Function to extract JSON data from the output string
def extract_json_from_output(output_string):
    # Use regex to find the JSON part of the string
    json_match = re.search(r'\[.*\]', output_string, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            return None
    else:
        print("No JSON found in the output.")
        return None


# Function to create instructions for humanizing AI-generated text
def create_instructions(ai_text):
    prompt = f"""
Humanize the AI-generated text so it bypasses ALL AI detectors! Identify all giveaways in the AI text, focusing on factors like perplexity, burstiness, tone, variation, repetition, etc. Provide specific, detailed instructions to make the AI text indistinguishable from human writing.

For each identified issue, provide:

- giveaway: A brief description of the issue (e.g., "Repetition")
- location: The location of the issue in the text (e.g., "Sentence 3")
- instructions: Detailed instructions on how to fix the issue, including specific examples and changes.
Do NOT include anything directly from the human text. Only provide instructions on modifying the AI text.

AI-generated text:
{ai_text}

Provide the output as a JSON array of objects:
[{{
"giveaway": "",
"location": "",
"instructions": ""
}}]
"""
    response = client.chat.completions.create(
        model=instructions_model_name,
        messages=[
            {"role": "system", "content": "You are an expert AI detector/humanizer who identifies ALL the AI generated elements from a text and suggests detailed methods to convert it into humanized text so it bypasses ALL AI detectors!"},
            {"role": "user", "content": prompt}
        ]
    )

    instructions = response.choices[0].message.content.strip()
    instructions = extract_json_from_output(instructions)
    return instructions


# Function to humanize AI-generated text based on detailed instructions
def fix_issues(instructions, ai_text):
    prompt = f"""
Given the AI-generated text from a research paper and the detailed instructions below, modify the AI text to HUMANIZE IT and make it INDISTINGUISHABLE from human-written text. Focus on addressing each identified issue by following the specific instructions provided. Make sure to Humanize the text to bypass ALL AI detectors!

Follow these guidelines:
1) Make the output approximately the same length as the input
2) Maintain the context and meaning of the text. Do not randomly "describe the research paper" or etc... unless the source text does!
3) INCLUDE THE CITATIONS AS IS! DO NOT change or remove existing citations and sources!!! 

AI-generated text:
"{ai_text}"

Instructions:
"{json.dumps(instructions, indent=4)}"
"""

    response = client.chat.completions.create(
        model=humanizer_model_name,
        messages=[
            {"role": "system", "content": """You are an expert in HUMANIZING AI-generated paragraphs WITHIN A RESEARCH PAPER to make it INDISTINGUISHABLE from human writing. Think of how a human would naturally write, ensuring the text flows smoothly, naturally, and engagingly. Focus on creating a varied and dynamic sentence structure, incorporating idiomatic expressions, and ensuring contextual relevance. Pay special attention to factors like perplexity, burstiness, tone, variation, and repetition."""},
            {"role": "user", "content": prompt}
        ]
    )

    humanized_text = response.choices[0].message.content.strip()
    return humanized_text
