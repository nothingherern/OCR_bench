from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import glob
from pathlib import Path
model_path = "microsoft/Phi-4-multimodal-instruct"

langs_dict = {
    "ja": "Japanese",
    "ru": "Russian",
    "en": "English",
    "cn": "Chinese",
    "ko": "Korean"
}

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
    attn_implementation='flash_attention_2',
).cuda()
generation_config = GenerationConfig.from_pretrained(model_path)
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

def test(langs, unOCRed_image_paths):
    result = []
    for lang in langs:
        print(f"Checking {lang}")
        images = glob.glob(f"test_data/{lang}/*")
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            file_name = Path(path).stem
            predicted_text = get_ocr_text(path, lang)
            print(predicted_text)

            data = (file_name, predicted_text, lang)
            result.append(data)
    return result

def get_ocr_text(path, lang):
    #Show me only text from image, nothing else, translated to Russian.
    prompt = f'{user_prompt}<|image_1|>Give me text from image, writen in {langs_dict[lang]} language, nothing else.{prompt_suffix}{assistant_prompt}'
    print(f'>>> Prompt\n{prompt}')
    image = Image.open(path)
    inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response