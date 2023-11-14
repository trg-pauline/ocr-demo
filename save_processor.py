from transformers import TrOCRProcessor
import json

print("Start saving")

model_name = 'microsoft/trocr-base-printed'
processor = TrOCRProcessor.from_pretrained(model_name)
processor.save_pretrained("processor")

print("Finished saving")