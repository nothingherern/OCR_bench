from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import glob
from pathlib import Path

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')


def test(langs, unOCRed_image_paths):
    result = []
    for lang in langs:
        print(f"Checking {lang}")
        images = glob.glob(f"test_data/{lang}/*")
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            file_name = Path(path).stem

            img = Image.open(path).convert("RGB")
            pixel_values = processor(images=img, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            data = (file_name, predicted_text, lang)
            print(data)
            result.append(data)
    return result
