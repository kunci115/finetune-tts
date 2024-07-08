import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# Configuration settings
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

# Configure the model with PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

# Update tokenizer with chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
)

# Formatting function for prompts
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# Load dataset and apply formatting
dataset = load_dataset("philschmid/guanaco-sharegpt-style", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs"
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args
)

# Train the model
trainer_stats = trainer.train()

FastLanguageModel.for_inference(model)

# Function to generate and print output
def generate_and_print(prompt):
    messages = [{"from": "human", "value": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)

# Generate outputs
generate_and_print("Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,")
generate_and_print("What is a famous tall tower in Paris?")

# Safe save function
def safe_save(save_func, *args, **kwargs):
    try:
        save_func(*args, **kwargs)
        print(f"Successfully saved using {save_func.__name__}")
    except Exception as e:
        print(f"Error when saving with {save_func.__name__}: {str(e)}")

# Move model to CPU for saving
model = model.cpu()

# Save functions
print("Starting save process...")

# Save LoRA weights
safe_save(model.save_pretrained_merged, "llama3-finetuned-conversational", tokenizer, save_method="lora")
safe_save(model.push_to_hub_merged, "kunci115/llama3-finetuned-conversational", tokenizer, save_method="lora", token="")

# Save GGUF format (8-bit quantization)
safe_save(model.save_pretrained_gguf, "llama3-finetuned-conversational", tokenizer)
safe_save(model.push_to_hub_gguf, "kunci115/llama3-finetuned-conversational", tokenizer, token="")

# Save GGUF format (4-bit quantization)
safe_save(model.save_pretrained_gguf, "llama3-finetuned-conversational", tokenizer, quantization_method="q4_k_m")
safe_save(model.push_to_hub_gguf, "kunci115/llama3-finetuned-conversational", tokenizer, quantization_method="q4_k_m", token="")

# If you're certain about 4-bit merged saving (use cautiously)
# safe_save(model.save_pretrained_merged, "llama3-finetuned-conversational", tokenizer, save_method="merged_4bit_forced")
# safe_save(model.push_to_hub_merged, "kunci115/llama3-finetuned-conversational", tokenizer, save_method="merged_4bit_forced", token="")

print("Saving process completed.")
