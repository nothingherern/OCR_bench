import easyocr
import glob
from pathlib import Path

langs_dict = {
    "ja": "ja",
    "ru": "ru",
    "en": "en",
    "cn": "ch_sim",
    "ko": "ko"
}

def test(langs, unOCRed_image_paths):
    result = []
    for lang in langs:
        reader = easyocr.Reader([langs_dict[lang]])
        print(f"Checking {lang}")
        images = glob.glob(f"test_data/{lang}/*")
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            file_name = Path(path).stem
            arr = reader.readtext(path, detail = 0)
            predicted_text = ''.join(str(x) for x in arr) 
            print(predicted_text)
            data = (file_name, predicted_text, lang)
            result.append(data)
    return result

# reader = easyocr.Reader(['en', 'ja'])
# result = reader.readtext('test_small.png',detail = 0)
# print(result)