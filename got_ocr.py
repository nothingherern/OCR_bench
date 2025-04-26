from transformers import AutoProcessor, AutoModelForImageTextToText
import glob
from pathlib import Path

device = "cuda"# if torch.cuda.is_available() else "cpu"
model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

def test(langs, unOCRed_image_paths):
    result = []
    for lang in langs:
        print(f"Checking {lang}")
        images = glob.glob(f"test_data/{lang}/*")
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            print(path)
            file_name = Path(path).stem

            predicted_text = get_ocr_text(path, lang)

            print("predicted_text")
            print(predicted_text)
            data = (file_name, predicted_text, lang)
            result.append(data)
    return result

def get_ocr_text(path, lang):
    inputs = processor(path, return_tensors="pt").to(device)

    generate_ids = model.generate(
        **inputs,
        do_sample=False,
        tokenizer=processor.tokenizer,
        stop_strings="<|im_end|>",
        max_new_tokens=4096,
    )

    result = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return result