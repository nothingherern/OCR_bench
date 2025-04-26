import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import glob
from pathlib import Path

langs_dict = {
    "ja": "Japanese",
    "ru": "Russian",
    "en": "English",
    "cn": "Chinese",
    "ko": "Korean"
}

# load omni model default, the default init_vision/init_audio/init_tts is True
# if load vision-only model, please set init_audio=False and init_tts=False
# if load audio-only model, please set init_vision=False
model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-o-2_6',
    trust_remote_code=True,
    attn_implementation='flash_attention_2', # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=False,
    init_tts=False
)


model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

def test(langs, unOCRed_image_paths):
    result = []
    print(langs)
    for lang in langs:
        print(f"Checking {lang}")
        images = glob.glob(f"test_data/{lang}/*")
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            file_name = Path(path).stem
            predicted_text = get_ocr_text(path, lang)
            data = (file_name, predicted_text, lang)
            result.append(data)
    return result

def get_ocr_text(path, lang):
    image = Image.open(path).convert('RGB')
    question = f"Give me text from image, writen in {langs_dict[lang]} language, nothing else."
    msgs = [{'role': 'user', 'content': [image, question]}]
    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )
    print(res)
    return res