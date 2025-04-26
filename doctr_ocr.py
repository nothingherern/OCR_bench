from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import glob
from pathlib import Path

def test(langs, unOCRed_image_paths):
    result = []
    model = ocr_predictor(pretrained=True)
    for lang in langs:
        print(f"Checking {lang}")
        images = glob.glob(f"test_data/{lang}/*")
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            print(path)
            file_name = Path(path).stem

            doc = DocumentFile.from_images(path)
            predicted_text = model(doc)

            print("predicted_text")
            print(predicted_text)
            data = (file_name, predicted_text, lang)
            result.append(data)
    return result

# model = ocr_predictor(pretrained=True)
# # PDF
# doc = DocumentFile.from_images("images/test2.png")
# # Analyze
# result = model(doc)
# #result.show()
# print(result)