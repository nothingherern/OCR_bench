import ddddocr
import glob
from pathlib import Path
def test(langs, unOCRed_image_paths):
    result = []
    ocr = ddddocr.DdddOcr(beta=True)
    for lang in langs:
        print(f"Checking {lang}")
        images = glob.glob(f"test_data/{lang}/*")
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            file_name = Path(path).stem

            image = open(path, "rb").read()
            predicted_text = ocr.classification(image)

            print("predicted_text")
            print(predicted_text)
            data = (file_name, predicted_text, lang)
            result.append(data)
    return result



# ocr = ddddocr.DdddOcr(beta=True)  # 切换为第二套ocr模型

# image = open("captcha2.png", "rb").read()
# result = ocr.classification(image)
# print("RES")
# print(result)