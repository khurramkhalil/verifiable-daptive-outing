from transformers import AutoConfig

model_name = "mistralai/Mixtral-8x7B-v0.1"
config = AutoConfig.from_pretrained(model_name)

print(f"Model: {model_name}")
print(f"Num Experts: {getattr(config, 'num_local_experts', 'Unknown')}")
print(f"Experts Per Token (K): {getattr(config, 'num_experts_per_tok', 'Unknown')}")
