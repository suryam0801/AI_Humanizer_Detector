import json
import requests
import random

# Path to the input JSON file
json_file_path = './outputs/training_output.json'
# Path to the output JSON file
output_file_path = 'results.json'

# Winston AI API configuration
winston_url = "https://api.gowinston.ai/functions/v1/predict"
winston_headers = {
    "Authorization": "JdOR4xakGZDX0SWS0VxIQuTUGlN2fjnij37NclEgb0bfdd6b",
    "Content-Type": "application/json"
}

# Function to read data from the JSON file


def read_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

# Function to write data to the JSON file


def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Function to get a random sample of entries


def get_random_sample(data, sample_size=200):
    return random.sample(data, sample_size)

# Function to call the Winston AI API and get the response


def get_winston_score(text):
    payload = {"text": text}
    response = requests.post(winston_url, json=payload,
                             headers=winston_headers)
    return response.json()

# Main function to process the data, get scores, and save results


def main():
    data = read_json(json_file_path)
    sample_data = get_random_sample(data, 200)
    results = []

    for entry in sample_data:
        text = entry.get("text", "")
        api_response = get_winston_score(text)
        result = {
            "input": text,
            "score": api_response.get("score", ""),
            "sentence_scores": api_response.get("sentences", [])
        }
        results.append(result)

    write_json(output_file_path, results)
    print(
        f"Processed {len(results)} entries and saved the results to '{output_file_path}'.")


if __name__ == "__main__":
    main()
