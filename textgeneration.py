from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch


dataset = load_dataset('text', data_files={'train': 'path/to/your/train.txt', 'test': 'path/to/your/test.txt'})

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


model = GPT2LMHeadModel.from_pretrained('gpt2')


training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=200,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
)


trainer.train()


model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')


from transformers import pipeline

generator = pipeline('text-generation', model='./fine_tuned_model')

prompt = "Once upon a time"
result = generator(prompt, max_length=50, num_return_sequences=1)
print(result)
