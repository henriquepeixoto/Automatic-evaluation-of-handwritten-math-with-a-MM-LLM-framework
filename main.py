
# Google Generative AI
import google.generativeai as genai

# Transformers and PyTorch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MllamaForConditionalGeneration,
    AutoProcessor
)
import torch

# OpenAI API
from openai import OpenAI

# Image processing and display
from PIL import Image, ImageDraw
from IPython.display import display
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# File and data handling
import os
import json
import pandas as pd
import numpy as np
from io import BytesIO
import re
import sys
from preprocess import Preprocess
from posprocess import Posprocess

class LLM_math_evaluator:
  def __init__(self, model_name, key, verbose=False):

    # load prompts
    self.SYSTEM_INSTRUCTIONS_1, self.SYSTEM_INSTRUCTIONS_2 = self.load_prompts()
    self.verbose = verbose

    # load preprocess class
    self.preprocessor = Preprocess(verbose=self.verbose)
    self.posprocessor = Posprocess()

    # load model
    if model_name == "gemini-1.5-pro":
        self.model_gemini = genai.GenerativeModel(model_name=model_name)
        genai.configure(api_key=key)
    elif model_name == "llama-3.2-11b":
        os.environ["HF_TOKEN"] = key

        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        self.model_llama = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.processor_llama = AutoProcessor.from_pretrained(model_id)
    elif model_name in ["gpt-4o-mini", "gpt-4o"]:
        os.environ["OPENAI_API_KEY"] = key
        self.model_openai = model_name
        self.client_openai = OpenAI(api_key=key)
    else:
        raise ValueError(f"Model {model_name} not supported, supported models are gemini-1.5-pro, llama-3.2-11b, gpt-4o-mini and gpt-4o")


  def load_prompts(self):

    SYSTEM_INSTRUCTIONS_1 = """You are an automated math solution evaluator. You will be provided with the following:
    1. A math question (optional, for context).
    2. An image containing a handwritten answer provided by a student.
    3. The correct answer provided by the teacher in LaTeX format.

    Your task is to evaluate whether the student’s final result in the image is correct.

    Key Points:
      1. Strictly compare the student's answer to the provided correct answer.
      2. If the handwritten answer matches the correct answer, output True.
      3. If the handwritten answer does NOT match the correct answer, output False.
      4. Do your best to interpret the handwriting in the image, but if it's completely illegible, err on the side of caution and output Erro.

    IMPORTANT:
    - DO NOT provide explanations, reasoning, or any text other than this JSON object.
    - Output ONLY the JSON object.
    {
    "correct": True or False,
    "student_answer": "the student's answer"
    }
    """

    SYSTEM_INSTRUCTIONS_2 =  """You are an automated math process evaluator. You will be provided with:
    1. A math question (optional, for context).
    2. A step-by-step solution provided by the teacher in LaTeX format.
    3. An image containing a handwritten answer provided by a student.

    Your task is to evaluate how closely the steps in the student’s handwritten solution (in the image) adhere to the logical structure and process of the teacher’s solution.

    Key Evaluation Points:
      1. Adherence Scoring (0 to 10):
          - Assign a score from 0 to 10 based on the overall similarity of the student's process to the teacher's process.
          - Higher scores indicate closer adherence.
          - Consider the following general guidelines:
              - 10: Identical process.
              - 9-8: Minor differences, but logically equivalent.
              - 7-6: Mostly correct, with a few small deviations.
              - 5-4: Some correct elements, but significant deviations.
              - 3-2: Limited understanding, major deviations or gaps.
              - 1-0: Unrelated, illogical, or nonsensical.
      2. Error Identification:
          - If the student’s solution deviates, identify the error types present.
          - Do not specify step numbers.
          - Choose from the following error types:

              - **SKIPPED_STEP:** The student skipped a necessary step from the teacher's solution.
              - **INCORRECT_ORDER:** The student performed steps in the wrong order.
              - **WRONG_METHOD:** The student used a different method or formula not shown in the teacher's solution.
              - **LOGICAL_ERROR:** The student made a logical fallacy or inconsistency in their reasoning.
              - **CALCULATION_ERROR:** While the process might be correct, the student made a calculation mistake (use sparingly, focus on process errors).
              - **ILLEGIBLE:** The handwriting is too illegible to interpret the process.

    IMPORTANT:
    - DO NOT provide explanations, reasoning, or any text other than this JSON object.
    - Output ONLY the JSON object.
    {
      "solution_adherence": Integer (0-10),
      "error_identification": ["SKIPPED_STEP", "INCORRECT_ORDER", ...]
    }
    """

    return SYSTEM_INSTRUCTIONS_1, SYSTEM_INSTRUCTIONS_2


  def openAI_inference(self, question, solution, answer, image_path, model, print_img=False):
    img = self.preprocessor.img_preprocess(img_path=image_path, resize=True, max_width=500, max_height=500)
    base64_image = self.preprocessor.encode_image(img)

    input1 = f"""
    Question: {question}
    Teacher's correct answer (LaTeX): {answer}
    """
    input2 = f"""
    Question: {question}
    Teacher's correct solution (LaTeX): {solution}
    """

    # Create the payload for the OpenAI API
    messages_payload_1 = [
        {"role": "system", "content": self.SYSTEM_INSTRUCTIONS_1},
        {"role": "user", "content": input1},
        {"role": "user", "content": f"data:image/png;base64,{base64_image}"}
    ]

    messages_payload_2 = [
        {"role": "system", "content": self.SYSTEM_INSTRUCTIONS_2},
        {"role": "user", "content": input2},
        {"role": "user", "content": f"data:image/png;base64,{base64_image}"}
    ]

    # Send the request to OpenAI
    response_1 = self.client_openai.chat.completions.create(
        model=self.model_openai,
        messages=messages_payload_1,
        temperature=0.0,
    )

    response_2 = self.client_openai.chat.completions.create(
        model=self.model_openai,
        messages=messages_payload_2,
        temperature=0.0,
    )

    if print_img:
        display(img)

    resp1 = re.sub(r'^```json\s*|\s*```$', '', response_1.choices[0].message.content, flags=re.DOTALL)
    resp2 = re.sub(r'^```json\s*|\s*```$', '', response_2.choices[0].message.content, flags=re.DOTALL)

    # Clean up the strings
    resp1 = self.posprocessor.clean_json_string(json_string=resp1)
    resp2 = self.posprocessor.clean_json_string(json_string=resp2)

    try:
        # Parse each JSON string into Python dictionaries
        parsed_resp1 = json.loads(resp1)
        parsed_resp2 = json.loads(resp2)

        # Merge the dictionaries
        merged_json = {**parsed_resp1, **parsed_resp2}

        # Print or use the merged JSON
        print(parsed_resp2)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e.msg}")
        print(f"Error at line {e.lineno}, column {e.colno}")
        print(f"Error document: {e.doc}")

    return merged_json

  def gemini_inference(self, question, solution, answer, image_path, print_img=False):
    input1 = f"""
    Question: {question}
    Teacher's correct answer (LaTeX): {answer}
    """
    input2 = f"""
    Question: {question}
    Teacher's correct solution (LaTeX): {solution}
    """

    img = self.preprocessor.img_preprocess(image_path)
    # For SYSTEM_INSTRUCTIONS_1
    response_1 = self.model_gemini.generate_content(
        [self.SYSTEM_INSTRUCTIONS_1, input1, img]
    )

    # For SYSTEM_INSTRUCTIONS_2
    response_2 = self.model_gemini.generate_content(
        [self.SYSTEM_INSTRUCTIONS_2, input2, img]
    )

    if print_img:
        display(img)

    resp1 = re.sub(r'^```json\s*|\s*```$', '', response_1.text, flags=re.DOTALL)
    resp2 = re.sub(r'^```json\s*|\s*```$', '', response_2.text, flags=re.DOTALL)

    # Clean up the strings
    resp1 = self.posprocessor.clean_json_string(json_string=resp1)
    resp2 = self.posprocessor.clean_json_string(json_string=resp2)

    try:
        # Parse each JSON string into Python dictionaries
        parsed_resp1 = json.loads(resp1)
        parsed_resp2 = json.loads(resp2)

        # Merge the dictionaries
        merged_json = {**parsed_resp1, **parsed_resp2}

        # Print or use the merged JSON
        print(merged_json)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e.msg}")
        print(f"Error at line {e.lineno}, column {e.colno}")
        print(f"Error document: {e.doc}")

    return merged_json

  def llama_inference(self, question, solution, answer, image_path, print_img=False):
    img = self.preprocessor.img_preprocess(image_path)

    if print_img:
        display(img)

    prompt_1 = f"""
    {self.SYSTEM_INSTRUCTIONS_1}
    Question: {question}
    Teacher's correct answer (LaTeX): {answer}
    Image:
    """
    prompt_2 = f"""
    {self.SYSTEM_INSTRUCTIONS_2}
    Question: {question}
    Teacher's correct solution (LaTeX): {solution}
    Image:
    """

    messages_1 = [
    {"role": "user", "content": [
        {"type": "text", "text": prompt_1},
        {"type": "image"},

      ]}
    ]

    messages_2 = [
    {"role": "user", "content": [
        {"type": "text", "text": prompt_2},
        {"type": "image"},
      ]}
    ]

    input_text_1 = self.processor_llama.apply_chat_template(messages_1, add_generation_prompt=False)
    input_text_2 = self.processor_llama.apply_chat_template(messages_2, add_generation_prompt=False)


    inputs_1 = self.processor_llama(
        img,
        input_text_1,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(self.model_llama.device)

    inputs_2 = self.processor_llama(
        img,
        input_text_2,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(self.model_llama.device)

    output_1 = self.model_llama.generate(**inputs_1, max_new_tokens=500)
    output_2 = self.model_llama.generate(**inputs_2, max_new_tokens=500)

    response_1 = self.processor_llama.decode(output_1[0], skip_special_tokens=True)
    response_2 = self.processor_llama.decode(output_2[0], skip_special_tokens=True)


    try:
        # Parse and clean responses
        parsed_resp1 = self.posprocessor.extract_and_transform_response_llama(response_1, "response_1")
        parsed_resp2 = self.posprocessor.extract_and_transform_response_llama(response_2, "response_2")


        # Merge the dictionaries
        merged_json = {**parsed_resp1, **parsed_resp2}

        # Print or use the merged JSON
        print(merged_json)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e.msg}")
        print(f"Error at line {e.lineno}, column {e.colno}")
        print(f"Error document: {e.doc}")

    return merged_json
