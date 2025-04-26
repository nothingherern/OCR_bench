import torch
from transformers import AutoModel, AutoTokenizer
import glob
from pathlib import Path

langs_dict = {
    "ja": "Japanese",
    "ru": "Russian",
    "en": "English",
    "cn": "Chinese",
    "ko": "Korean"
}

# Set up the model and tokenizer
model_path = 'h2oai/h2ovl-mississippi-2b'
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=1024, do_sample=True)

def test(langs, unOCRed_image_paths):
    result = []
    print(langs)
    for lang in langs:
        print(f"Checking {lang}")
        images = glob.glob(f"test_data/{lang}/*")
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            file_name = Path(path).stem
            predicted_text = get_ocr_text(path, lang)
            data = (file_name, predicted_text, lang)
            result.append(data)
    return result
def get_ocr_text(path, lang):
    # Example for single image
    image_file = path
    question = f'<image>\nGive me text from image, writen in {langs_dict[lang]} language, nothing else.'
    response, history = model.chat(tokenizer, image_file, question, generation_config, history=None, return_history=True)
    print(response)
    return response
