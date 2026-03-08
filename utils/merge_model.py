from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

adapter_path = "/data/jonathan/Lost-in-Mistranslation/models/olmo2_klar_lora/final_adapter"
output_path = "/data/jonathan/Lost-in-Mistranslation/models/olmo2_klar_lora_full"

# Read base model name from the adapter config
peft_config = PeftConfig.from_pretrained(adapter_path)
base_model_name = peft_config.base_model_name_or_path

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,   # or torch.float16
    device_map="auto",
)

# Load adapter on top
model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge LoRA into base weights
merged_model = model.merge_and_unload()

# Save merged standalone model
merged_model.save_pretrained(output_path, safe_serialization=True)

# Save tokenizer too
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(output_path)

print(f"Merged model saved to {output_path}")