from manga_ocr import MangaOcr
import glob
from pathlib import Path
def test(langs, unOCRed_image_paths):
    result = []
    mocr = MangaOcr()
    for lang in langs:
        print(f"Checking {lang}")
        images = glob.glob(f"test_data/{lang}/*")
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            file_name = Path(path).stem
            predicted_text = mocr(path)
            data = (file_name, predicted_text, lang)
            result.append(data)
    return result

    