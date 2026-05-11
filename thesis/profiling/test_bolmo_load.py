from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_path = os.path.expandvars("$VSC_SCRATCH/master_thesis/checkpoints/Bolmo-1B")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Model path:", model_path)
print("Device:", device)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
).to(device)

prompt = ["Language modeling is "]
inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=32,
        do_sample=False,
    )

print("\nOUTPUT:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))