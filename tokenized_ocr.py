from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import glob
from pathlib import Path

langs_dict = {
    "ja": "Japanese",
    "ru": "Russian",
    "en": "English",
    "cn": "Chinese",
    "ko": "Korean"
}

# Load the Tokenized-OCR model with optimized parameters
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/Tokenized-OCR", torch_dtype="auto", device_map="auto"
)

# Recommended acceleration for performance optimization:
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "prithivMLmods/Tokenized-OCR",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# Load the default processor for Tokenized-OCR
processor = AutoProcessor.from_pretrained("prithivMLmods/Tokenized-OCR")

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
    #prompt = f"Give me text from image, writen in {langs_dict[lang]} language, nothing else."
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": path,
                },
                {"type": "text", "text": "Extract and return the tokenized OCR text from the image, ensuring each word is accurately recognized and separated by commas."},
            ],
        }
    ]

    # Prepare the input for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    result = output_text[0]
    #print("prompt: " + prompt)
    print("output_text: " + result)
    return result