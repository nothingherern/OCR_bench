from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch
import glob
from pathlib import Path

#from huggingface_hub import login
#login()

langs_dict = {
    "ja": "Japanese",
    "ru": "Russian",
    "en": "English",
    "cn": "Chinese",
    "ko": "Korean"
}

model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)

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

def get_ocr_text(path, lang):
    image = Image.open(path).convert("RGB")

    prompt = f"<image>Give me text from image, writen in {langs_dict[lang]} language, nothing else."
    model_inputs = processor(text=prompt, images=image, return_tensors="pt")
    input_len = model_inputs["input_ids"].shape[-1]

    result = ""
    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        result = decoded
    print(result)

    return result
    