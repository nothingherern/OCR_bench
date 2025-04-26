import winocr
from winocr import recognize_cv2_sync
import cv2
import glob
from pathlib import Path
import asyncio
from PIL import Image


# Get all langs command for powershell
##  Get-WindowsCapability -Online | Where-Object { $_.Name -Like 'Language.OCR*' }
# List of powershell commands to install langs
##  Add-WindowsCapability -Online -Name "Language.OCR~~~ja-JP~0.0.1.0"
##  Add-WindowsCapability -Online -Name "Language.OCR~~~ru-RU~0.0.1.0"
##  Add-WindowsCapability -Online -Name "Language.OCR~~~en-US~0.0.1.0"
##  Add-WindowsCapability -Online -Name "Language.OCR~~~zh-CN~0.0.1.0"
##  Add-WindowsCapability -Online -Name "Language.OCR~~~ko-KR~0.0.1.0"
langs_dict = {
    "ja": "ja-JP",
    "ru": "ru-RU",
    "en": "en-US",
    "cn": "zh-CN",
    "ko": "ko-KR"
}
async def to_coroutine(awaitable):
    return await awaitable

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
            #img = cv2.imread(path)
            #print(langs_dict[lang])
            
            predicted_text = asyncio.run(to_coroutine(winocr.recognize_pil(img, langs_dict[lang]))).text
            #predicted_text = recognize_cv2_sync(img, lang=langs_dict[lang])['text']
            print(predicted_text)
            data = (file_name, predicted_text, lang)
            result.append(data)
    return result
