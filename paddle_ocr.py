from paddleocr import PaddleOCR, draw_ocr
import glob
from pathlib import Path

# Paddleocr supports Chinese, English, French, German, Korean and Japanese
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order

langs_dict = {
    "ja": "japan",
    "ru": "ru",
    "en": "en",
    "cn": "ch",
    "ko": "korean"
}

def test(langs, unOCRed_image_paths):
    result = []
    for lang in langs:
        print(f"Checking {lang}")
        images = glob.glob(f"test_data/{lang}/*")
        ocr = PaddleOCR(use_angle_cls=True, lang=langs_dict[lang])
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            file_name = Path(path).stem
            ocr_output = ocr.ocr(path, cls=True)
            predicted_text = ""
            if len(ocr_output) == 0:
                predicted_text = ""
            else:
                for idx in range(len(ocr_output)):
                    # print("ocr_output")
                    # print(ocr_output)
                    res = ocr_output[idx]
                    if res == None:
                        predicted_text += ""
                    else:
                        for data in res:
                            predicted_text += data[1][0]
            print("--predicted_text--")
            print(predicted_text)
            print("-------------------")

            data = (file_name, predicted_text, lang)
            result.append(data)
    return result

# ocr = PaddleOCR(use_angle_cls=True, lang='japan') # need to run only once to download and load model into memory
# img_path = 'images/test_small2.png'
# result = ocr.ocr(img_path, cls = True)
# print("result")
# for idx in range(len(result)):
#     res = result[idx]
#     for data in res:
#         print(data[1][0])