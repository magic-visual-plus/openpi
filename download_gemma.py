import torch
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
from modelscope import snapshot_download, AutoModel, AutoTokenizer

model_id = "Qwen/Qwen2.5-14B-Instruct"
model_id = "google/paligemma-3b-pt-224"
# model_id = "google/paligemma-3b-mix-224"

# model_dir = snapshot_download(model_id, cache_dir='/opt/models/', revision='master')

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch

# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(model_id, token="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")
model = AutoModelForImageTextToText.from_pretrained(model_id, token="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")