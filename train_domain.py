from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset
import os

# Ensure directories exist
directories = ['./model', './logs', './train']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  

# Load your dataset using the Datasets library
dataset = load_dataset('text', data_files={'train': './train/train.txt', 'validation': './train/valid.txt'})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    # Model Saving and Loading
    output_dir="./model",
    overwrite_output_dir=True,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    save_steps=100,
    save_total_limit=3,
    # Dataset and Evaluation
    num_train_epochs=20,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    eval_steps=100,
    greater_is_better=False,
    # Learning Rate, Optimization, and Regularization
    learning_rate=1e-5,
    lr_scheduler_type="linear",#newly added to attempt to refine the learning rate
    weight_decay=0.01, #newly added to attempt to refine the learning rate
    gradient_accumulation_steps=2,
    max_grad_norm=0.5,
    warmup_steps=20,
    # Logging
    logging_first_step=True,
    logging_dir="./logs",
    logging_steps=50,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)],
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model()
tokenizer.save_pretrained("./model")