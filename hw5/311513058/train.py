from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np
import json



dataset = load_dataset("json", data_files="hw5_dataset/train.json")
dataset=dataset['train'].train_test_split(test_size=0.2)


checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)




def preprocess_function(data):
    input = ["summarize: " + token for token in data["body"]]
    input_id = tokenizer(input, max_length=512, truncation=True)

    label = tokenizer(text_target=data["title"], max_length=128, truncation=True)

    input_id["labels"] = label["input_ids"]
    return input_id

dataset=dataset.filter(lambda x : x['body'] is not None)
tokenized_dataset =dataset.map(preprocess_function, batched=True)



data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)



rouge = evaluate.load("rouge",rouge_types=["rouge1", "rouge2", "rougeL"])
metric_bertscore = evaluate.load("bertscore")




def compute_metrics(pred):
    predic_id, label_id = pred
    predic_token = tokenizer.batch_decode(predic_id, skip_special_tokens=True)
    
    label_id = np.choose(label_id != -100, [tokenizer.pad_token_id, label_id])

    label_token = tokenizer.batch_decode(label_id, skip_special_tokens=True)

    result = rouge.compute(predictions=predic_token, references=label_token, use_stemmer=True)

    bertscore = metric_bertscore.compute(predictions=predic_token, references=label_token, lang="en")
    result["bertscore"] = np.mean(bertscore['precision'])

    eval_result={}
    for key,value in result.items():
        eval_result[f'{key}']=value
    return eval_result


model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="save",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=100,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    
)

trainer.train()


def test_preprocess_function(data):
     
    input = ["summarize: " + token if token is not None else "" for token in data["body"] ]
    input_id = tokenizer(input, max_length=512, truncation=True)

    return input_id


testdataset = load_dataset("json", data_files="hw5_dataset/test.json")['train']

tokenized_testdataset =testdataset.map(test_preprocess_function, batched=True)

raw_pred, _, _ = trainer.predict(tokenized_testdataset)


json_list = []


for title_id in raw_pred:
    json_list.append({"title": tokenizer.decode(title_id,skip_special_tokens=True)})



with open("311513058.json", "w") as f:
    for item in json_list:
        f.write(json.dumps(item) + "\n")