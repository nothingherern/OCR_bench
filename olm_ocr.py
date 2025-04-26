import torch
import base64
import urllib.request
import math
import json

from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

import cv2
import base64
import glob
from pathlib import Path

# Initialize the model
model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
device = "cuda"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_ocr_text(path, lang):
    # Grab image
    img = cv2.imread(path)
    height, width, channels = img.shape
    new_height = height * 0.75
    new_height_text = str(float(new_height))
    new_height_text = new_height_text[0:new_height_text.find(".")+2]
    new_width = width * 0.75
    new_width_text = str(float(new_width))
    new_width_text = new_width_text[0:new_width_text.find(".")+2]
    jpg_img = cv2.imencode('.png', img)
    b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')


    # Render page 1 to an image
    image_base64 = b64_string

    prompt = "Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. Just return the plain text representation of this document as if you were reading it naturally.\nDo not hallucinate.\n"
    prompt += f"RAW_TEXT_START\nPage dimensions: {new_width_text}x{new_height_text}\n[Image 0x0 to {str(math.ceil(new_width))}x{str(math.ceil(new_height))}]\n\nRAW_TEXT_END"

    #print("prompt: " + prompt)
    # Build the full prompt
    messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]

    # Apply the chat template and processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}

    # Generate the output
    output = model.generate(
                **inputs,
                temperature=0.8,
                max_new_tokens=200,
                num_return_sequences=1,
                do_sample=True,
            )

    # Decode the output
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )
    res = ""
    try:
        data = json.loads(text_output[0])
        res = data["natural_text"]
    except:
        print("Error with parsing json")
        print(text_output)
    if res == None:
        res = ""
    return res
    




def test(langs, unOCRed_image_paths):
    result = []
    for lang in langs:
        print(f"Checking {lang}")
        images = glob.glob(f"test_data/{lang}/*")
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            file_name = Path(path).stem
            predicted_text = get_ocr_text(path, lang)
            print(predicted_text)

            data = (file_name, predicted_text, lang)
            result.append(data)
    return result