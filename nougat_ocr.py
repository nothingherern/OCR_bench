from huggingface_hub import hf_hub_download
import re
from PIL import Image

from transformers import NougatProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import torch

processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# prepare PDF image for the model
#filepath = "images/test_en.png"
filepath = "images/test_en.png"
image = Image.open(filepath).convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values

# generate transcription (here we only generate 30 tokens)
outputs = model.generate(
    pixel_values.to(device),
    min_length=1,
    max_new_tokens=30,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)
# note: we're using repr here such for the sake of printing the \n characters, feel free to just print the sequence
print("result")
print(repr(sequence))