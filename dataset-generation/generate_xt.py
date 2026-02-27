import os, json
from litellm import completion
from system_prompt import SYSTEM_PROMPT as SYSTEM_PROMPT

MODEL = "azure/gpt5"
api_key = "api-key"

JSON_SCHEMA = {
    "name": "abstraction_steps_dataset",
    "schema": {
        "type": "object",
        "properties": {
            "response": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "x": {"type": "string"},
                        "xt": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                    },
                    "required": ["id", "x", "xt"],
                }
            } 
        },
        "required": ["response"],
        "additionalProperties": False  
    }
}

def get_user_input(sentence):
    user_input = f"Sentence: {sentence}"
    return user_input


def generate_abstractions(input_json_data) -> dict:
    response = completion(
        model="openai/gpt4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_json_data}
        ],
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA, "strict": True},
        api_base="https://thekeymaker.umass.edu/",
        api_key=api_key,
        temperature=0.2,
    )
    content = response.choices[0].message.content
    return json.loads(content)


def get_input_data(input_jsonl_file):
    with open(input_jsonl_file, "r") as f:
        jsonl_text = f.read()
    return jsonl_text


if __name__ == "__main__":
    input_jsonl_file = "../../input_x_10.jsonl"
    input_data = get_input_data(input_jsonl_file)
    # print(input_data)
    response = generate_abstractions(input_data)
    with open("../../output_x_10_v5.json", "w") as f:
        json.dump(response, f, indent=2)