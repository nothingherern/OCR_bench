from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import glob
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "ibm-granite/granite-vision-3.2-2b"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)

langs_dict = {
    "ja": "Japanese",
    "ru": "Russian",
    "en": "English",
    "cn": "Chinese",
    "ko": "Korean"
}

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
            print("predicted_text[0]")
            print(predicted_text)
            print("predicted_text[1]")

            data = (file_name, predicted_text, lang)
            result.append(data)
    return result
def get_ocr_text(path, lang):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": path},
                {"type": "text", "text": f"Give me text from image, writen in {langs_dict[lang]} language, nothing else."},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)
    result = processor.decode(output[0], skip_special_tokens=True)
    result = result[result.rfind("<|assistant|>")+len("<|assistant|>"):]
    result = result.strip()
    return result