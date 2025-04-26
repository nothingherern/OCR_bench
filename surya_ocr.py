from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
import glob
from pathlib import Path

recognition_predictor = RecognitionPredictor()
detection_predictor = DetectionPredictor()

langs_dict = {
    "ja": "ja",
    "ru": "ru",
    "en": "en",
    "cn": "zh",
    "ko": "ko"
}


def test(langs, unOCRed_image_paths):
    result = []
    for lang in langs:
        print(f"Checking {lang}")
        images = glob.glob(f"test_data/{lang}/*")
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            file_name = Path(path).stem
            image = Image.open(path)
            predictions = recognition_predictor([image], [[langs_dict[lang]]], detection_predictor)
            predicted_text = ""
            for prediction in predictions:
                for line in prediction.text_lines:
                    predicted_text += line.text + "\n"
            predicted_text = predicted_text.strip()
            print(predicted_text)
            data = (file_name, predicted_text, lang)
            result.append(data)
    return result

# predictions = recognition_predictor([image], [langs], detection_predictor)
# for prediction in predictions:
#     for line in prediction.text_lines:
#         print(line.text)