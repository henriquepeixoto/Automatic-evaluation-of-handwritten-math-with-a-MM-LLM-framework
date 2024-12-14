
# Regular expressions for string processing
import re

# JSON parsing
import json

class Posprocess:
  def __init__(self):
    pass
    
  def clean_json_string(self, json_string):
    # Replace Python-style booleans with JSON equivalents
    json_string = json_string.strip()
    json_string = json_string.replace("True", "true").replace("False", "false")
    # Replace commas in numbers (e.g., 0,3 -> 0.3)
    json_string = re.sub(r'(\d),(\d)', r'\1.\2', json_string)
    return json_string

  def try_fix_missing_quotes(self, s):
    if s.count('"') % 2 != 0 and s.strip().endswith('}'):
        s = s[:-1] + '"' + s[-1]
    return s

  def extract_and_transform_response_llama(self, response, response_type):

    json_block = re.search(r'assistant\s*(\{.*?\})', response, re.DOTALL)
    if not json_block:
        return None

    json_str = json_block.group(1)
    json_str = json_str.replace('\\', '\\\\')
    json_str = re.sub(r'\bTrue\b', 'true', json_str)
    json_str = re.sub(r'\bFalse\b', 'false', json_str)
    json_str = self.try_fix_missing_quotes(json_str)

    try:
        parsed_json = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to decode {response_type}: {json_str}")
        print(e)
        return None

    if response_type == "response_1":
        return {
            "correct": parsed_json.get("correct"),
            "student_answer": parsed_json.get("student_answer")
        }
    elif response_type == "response_2":
        return {
            "solution_adherence": parsed_json.get("solution_adherence"),
            "error_identification": parsed_json.get("error_identification")
        }
