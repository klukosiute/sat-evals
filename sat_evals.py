import pandas as pd
from openai import OpenAI
import os
import base64
import requests

# for multiple choice questions
GRADER_PROMPT = """Could you please help me grade my students' tests? I will show you a student's answer to a question and the correct answer(s). Please tell me whether the student answered the question correctly or incorrectly.
STUDENT ANSWER: {student_answer}
CORRECT ANSWER(S): {corrects}
Did the student answer the question correctly? Answer with YES if the student answered correctly or answer with NO if the student answered incorrectly. Do not provide any commentary. """


GPT4 = "gpt-4-1106-preview"
GPT3 = "gpt-3.5-turbo"


ALL_COLUMNS_POST_EVAL = [
    "calculator",
    "correct_answers",
    "is_multiple_choice",
    "contains_image",
    "question",
    "model_answer_text",
    "grader_answer_text",
    "model_correct",
]


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def run_vision_eval(
    client: OpenAI,
    api_key: str,
    grader_model: str,
    dataset: pd.DataFrame,
    cache_file_path: str,
):
    """Shows an image containing a question to GPT-4-V, model samples some text,
    another model sees the answer and reference answer(s), says whether the original model got the question correct.
    """

    if os.path.exists(cache_file_path):
        cache = pd.read_csv(cache_file_path)
    else:
        cache = pd.DataFrame(columns=ALL_COLUMNS_POST_EVAL)

    # starts from where it left off
    questions_to_process = pd.merge(
        dataset,
        cache,
        on="question",
        how="left",
        suffixes=("", "_cached"),
    )

    for i, row in questions_to_process.iterrows():

        if i % 10 == 0:
            print("Processing Question No. ", i)
        # Check if the question was not answered before
        if pd.isna(row["model_correct"]):
            image_path = os.path.join(
                "./data/oct_2023_sat_math_images", "q" + str(i + 1) + ".png"
            )
            base64_image = encode_image(image_path)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please answer the question given in the image. ",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 512,
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )

            model_answer_text = response.json()["choices"][0]["message"]["content"]

            corrects_text = " ".join(row["correct_answers"])
            grader_prompt = GRADER_PROMPT.format(
                student_answer=model_answer_text, corrects=corrects_text
            )

            grader_prompt = [{"role": "user", "content": grader_prompt}]

            grader_completion = client.chat.completions.create(
                model=grader_model,
                messages=grader_prompt,
                max_tokens=1,
                temperature=0,
                n=1,
            )

            grader_answer = grader_completion.choices[0].message.content.strip()

            model_correct = grader_answer.strip().startswith("Y")

            # Update cache with new response

            new_entry = {
                "calculator": row["calculator"],
                "correct_answers": row["correct_answers"],
                "is_multiple_choice": row["is_multiple_choice"],
                "contains_image": row["contains_image"],
                "question": row["question"],
                "model_answer_text": model_answer_text,
                "grader_answer_text": grader_answer,
                "model_correct": model_correct,
            }

            cache = pd.concat([cache, pd.DataFrame([new_entry])], ignore_index=True)
            # update cache
            cache.to_csv(cache_file_path, index=False)
        else:
            continue

    return cache


def run_text_eval(
    client: OpenAI,
    model: str,
    grader_model: str,
    dataset: pd.DataFrame,
    cache_file_path: str,
):
    """Shows a text-only model a question (multiple choice or open-text), model samples some text,
    another model sees the answer and reference answer(s), says whether the original model got the question correct.
    """

    if os.path.exists(cache_file_path):
        cache = pd.read_csv(cache_file_path)
    else:
        cache = pd.DataFrame(columns=ALL_COLUMNS_POST_EVAL)

    # starts from where it left off
    questions_to_process = pd.merge(
        dataset,
        cache,
        on="question",
        how="left",
        suffixes=("", "_cached"),
    )

    for i, row in questions_to_process.iterrows():
        if i % 5 == 0:
            print("Processing Question No. ", i)
        # Check if the question was not answered before
        if pd.isna(row["model_correct"]):

            message = [{"role": "user", "content": row["question"]}]

            completion = client.chat.completions.create(
                model=model,
                messages=message,
                n=1,
            )

            model_answer_text = completion.choices[0].message.content

            corrects_text = " ".join(row["correct_answers"])
            grader_prompt = GRADER_PROMPT.format(
                student_answer=model_answer_text, corrects=corrects_text
            )

            grader_prompt = [{"role": "user", "content": grader_prompt}]

            grader_completion = client.chat.completions.create(
                model=grader_model,
                messages=grader_prompt,
                max_tokens=1,
                temperature=0,
                n=1,
            )

            grader_answer = grader_completion.choices[0].message.content.strip()

            model_correct = grader_answer.strip().startswith("Y")

            # Update cache with new response

            new_entry = {
                "calculator": row["calculator"],
                "correct_answers": row["correct_answers"],
                "is_multiple_choice": row["is_multiple_choice"],
                "contains_image": row["contains_image"],
                "question": row["question"],
                "model_answer_text": model_answer_text,
                "grader_answer_text": grader_answer,
                "model_correct": model_correct,
            }

            cache = pd.concat([cache, pd.DataFrame([new_entry])], ignore_index=True)
            # update cache
            cache.to_csv(cache_file_path, index=False)
        else:
            continue

    return cache


def main():

    ## run the text eval on GPT-3
    with open("/Users/kamile/code/oai/.openai.secret", "r") as f:
        contents = f.read().strip()
    api_key = contents.split("\n")[0].split("=")[1].strip()
    client = OpenAI(api_key=api_key)

    dataset = pd.read_json("./data/oct_2023_sat_math.jsonl", lines=True)
    result = run_text_eval(client, GPT3, GPT3, dataset, "gpt-3-text-eval.cache")
    print("Accuracy GPT-3", sum(result["model_correct"]) / len(result))

    ## run the text eval on GPT-4
    result = run_text_eval(client, GPT4, GPT3, dataset, "gpt-4-text-eval.cache")
    print("Accuracy GPT-4", sum(result["model_correct"]) / len(result))

    ## run the vision eval (GPT-4 + vision)
    result = run_vision_eval(client, api_key, GPT3, dataset, "gpt-4-vision-eval.cache")
    print("Accuracy Vision GPT-4", sum(result["model_correct"]) / len(result))


if __name__ == "__main__":
    main()
