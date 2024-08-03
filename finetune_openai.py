from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key='')

# # Upload the training file
# response = client.files.create(
#     file=open("instructions_training_data_set.jsonl", "rb"),
#     purpose="fine-tune"
# )

# # Access the file ID from the response
# training_file_id = response.id
# print(f"Training file uploaded with ID: {training_file_id}")

# # Create a fine-tuning job
# response = client.fine_tuning.jobs.create(
#     training_file=training_file_id,
#     model="gpt-4o-mini-2024-07-18"
# )

# Access the fine-tuning job ID from the response
fine_tuning_job_id = "ftjob-2DnVAQFCnpxGmArsgIJiBJm0"  # response.id
print(f"Fine-tuning job created with ID: {fine_tuning_job_id}")

# # List 10 fine-tuning jobs
# response = client.fine_tuning.jobs.list(limit=10)
# print("List of fine-tuning jobs:")
# for job in response.data:
#     print(job)

# Retrieve the state of a specific fine-tuning job
job_id = fine_tuning_job_id  # Replace with your fine-tuning job ID
response = client.fine_tuning.jobs.retrieve(job_id)
print(f"Status of job {job_id}: {response.status}")

# Retrieve fine-tuned model name
fine_tuned_model_name = response.fine_tuned_model
print(f'The fine-tuned model name is: {fine_tuned_model_name}')

# # Cancel a fine-tuning job
# response = client.fine_tuning.jobs.cancel(job_id)
# print(f"Cancelled job {job_id}")

# # List up to 10 events from a fine-tuning job
# response = client.fine_tuning.jobs.list_events(
#     fine_tuning_job_id=job_id, limit=10)
# print(f"Events for job {job_id}:")
# for event in response.data:
#     print(event)

# # Delete a fine-tuned model (must be an owner of the org the model was created in)
# model_id = "ft:gpt-3.5-turbo:acemeco:suffix:abc123"  # Replace with your model ID
# response = client.models.delete(model_id)
# print(f"Deleted model {model_id}")

# Use the fine-tuned model
# Replace with your fine-tuned model name
# fine_tuned_model_name = "ft:gpt-4o-mini-2024-07-18:prosp::9rgTmhF3"
# response = client.chat.completions.create(
#     model=fine_tuned_model_name,
#     messages=[
#         {"role": "system", "content": """You are an expert in modifying AI-generated research paper text to make it indistinguishable from human writing. Think of how a human academic would naturally write, ensuring the text flows smoothly, naturally, and engagingly. Focus on creating varied and dynamic sentence structures, incorporating idiomatic expressions, and ensuring contextual relevance. ONLY return the modified sentence, nothing else. IF ANY CITATIONS ARE INCLUDED, RETURN THEM AS IS! DO NOT change or remove existing citations and sources."""},
#         {"role": "user", "content": """
# You are an expert in modifying AI-generated text to make it indistinguishable from human writing. Your task is to thoroughly analyze the following AI-generated paragraph, identify specific reasons that make it AI-generated, and provide a humanized version of the entire paragraph. Focus on the following key indicators:

# - Unnatural tone
# - Repetitive sentence structures
# - Lack of variation in word choice
# - Generalization without specific examples
# - Inconsistency in style and terminology
# - Lack of coherence or logical flow

# For each identified giveaway, describe the issue and how you corrected it. Ensure that the humanized paragraph flows naturally, incorporates varied sentence structures and word choices, and matches the tone and style of the provided human-written example.

# Here is the AI-generated paragraph to analyze and humanize:
# "Several key trends are redefining the future of work, including digitalization, connectivity, automation, AI, and shifts in workforce composition. The rise of digital technologies has led to non-stop connectivity, reshaping work dynamics and workflows. Automation and AI are transforming tasks previously undertaken by humans, increasing productivity but also posing challenges for organizational leaders. Workforce composition is also evolving, with significant shifts in skill requirements and demographic changes, such as an aging population and a multi-generational workforce[4][6]. These trends necessitate comprehensive policies in labor market regulation, education, and social welfare to navigate the changing work environment effectively."


# Here is a human-written paragraph for reference:
# "In this paper, a new text transformation technique called Semi-Adaptive Substitution Coder for Lossless Text Compression is proposed. The rapid advantage of this Substitution Coder is that it substitutes the codewords by referring the reference of the word's position in the dictionary to expedite the dictionary mapping and also codewords are shorter than words and, thus, the same amount of text will require less space. In general, text transformation needs an external dictionary to store the frequently used words. To preserve this transformation method in a healthy way, a semiadaptive dictionary is used and therefore which reduces the expenditure of memory overhead and speeds up the transformation because of the smaller size dictionary. This new transformation algorithm is implemented and tested using Calgary Corpus and Large Corpus. In this implementation Semi-Adaptive Substitution Coder in connection with a popular bzip2 and commonly used Gzip compressors improve the compression performance by about 7-9% on large files."

# Provide the humanized version of the AI-generated paragraph!
# """}
#     ]
# )
# print("Response from the fine-tuned model:")
# print(response.choices[0].message.content.strip())
