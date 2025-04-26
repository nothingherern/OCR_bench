from transformers import AutoProcessor, AutoModelForImageTextToText
import glob
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()

#Broken modeling_GOT.py file
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
    image_file = path

    # plain texts OCR
    res = model.chat(tokenizer, image_file, ocr_type='ocr')
    return res