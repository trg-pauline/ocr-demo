print("Start saving")

from transformers import TrOCRProcessor

model_name = 'microsoft/trocr-base-printed'
processor = TrOCRProcessor.from_pretrained(model_name)
processor.save_pretrained("processor")

print("Finished saving")