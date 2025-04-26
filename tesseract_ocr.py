from PIL import Image

import pytesseract
import glob
from pathlib import Path

langs_dict = {
    "ja": "jpn",
    "ru": "rus",
    "en": "eng",
    "cn": "chi_sim",
    "ko": "kor"
}

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def test(langs, unOCRed_image_paths):
    result = []
    for lang in langs:
        print(f"Checking {lang}")
        images = glob.glob(f"test_data/{lang}/*")
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            file_name = Path(path).stem
            img = Image.open(path)
            predicted_text = pytesseract.image_to_string(img, lang=langs_dict[lang])
            data = (file_name, predicted_text, lang)
            result.append(data)
    return result