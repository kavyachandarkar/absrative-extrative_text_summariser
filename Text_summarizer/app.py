from flask import Flask, request, jsonify, render_template
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from summarizer import Summarizer
import PyPDF2
import nltk
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset
from translate import Translator
import os

nltk.download('punkt')

app = Flask(__name__)

MODEL_DIR = "./my_finetuned_bart"

if os.path.exists(MODEL_DIR):
    tokenizer = BartTokenizer.from_pretrained(MODEL_DIR)
    abstractive_model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)
else:
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    abstractive_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

abstractive_summarizer = pipeline("summarization", model=abstractive_model, tokenizer=tokenizer)
extractive_summarizer = Summarizer()


def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    except Exception as e:
        return None
    return text.strip()


def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure
    }


def compute_bleu(reference, generated):
    reference_tokens = [sent_tokenize(reference)]
    generated_tokens = sent_tokenize(generated)
    return sentence_bleu(reference_tokens, generated_tokens) * 100


def compute_precision_recall(reference, generated):
    reference_words = set(word_tokenize(reference.lower()))
    generated_words = set(word_tokenize(generated.lower()))
    precision = len(generated_words.intersection(reference_words)) / len(generated_words) if generated_words else 0.0
    recall = len(generated_words.intersection(reference_words)) / len(reference_words) if reference_words else 0.0
    return {"Precision": precision, "Recall": recall}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    text = ""

    if request.is_json:
        data = request.get_json()
        text = data.get("text", "").strip()
        summary_type = data.get("type", "").lower()
        max_length = int(data.get("max_length", 130))
        min_length = int(data.get("min_length", 30))
    else:
        data = request.form
        summary_type = data.get("type", "").lower()
        max_length = int(data.get("max_length", 130))
        min_length = int(data.get("min_length", 30))

        if 'file' in request.files:
            pdf_file = request.files['file']
            text = extract_text_from_pdf(pdf_file)
            if not text:
                return jsonify({"error": "Failed to extract text from the PDF."}), 400
        else:
            text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    reference_summary = " ".join(sent_tokenize(text)[:3])

    if summary_type == "extractive":
        summary = extractive_summarizer(text, num_sentences=4)
        summary_text = " ".join(summary) if isinstance(summary, list) else summary

    elif summary_type == "abstractive":
        if len(text.split()) < 50:
            return jsonify({"error": "Text too short for abstractive summarization. Provide at least 50 words."}), 400
        summary = abstractive_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        summary_text = summary[0]['summary_text'] if summary else "Summarization failed."

    else:
        return jsonify({"error": "Invalid summary type. Use 'extractive' or 'abstractive'."}), 400

    rouge_scores = compute_rouge(reference_summary, summary_text)
    bleu_score = compute_bleu(reference_summary, summary_text)
    precision_recall = compute_precision_recall(reference_summary, summary_text)

    return jsonify({
        "summary": summary_text,
        "rouge_scores": rouge_scores,
        "bleu_score": bleu_score,
        "precision_recall": precision_recall
    })


# ✅ Added Hindi translation route here without changing anything else
@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text = data.get('text', '')
    translated = Translator(to_lang='hi').translate(text)
    return jsonify({'translated_text': translated})


@app.route('/train', methods=['POST'])
def train_model():
    params = request.get_json()

    train_path = params.get("train_path", "train.json")
    test_path = params.get("test_path", "test.json")
    num_train_epochs = params.get("epochs", 1)
    train_batch_size = params.get("train_batch_size", 2)
    eval_batch_size = params.get("eval_batch_size", 2)

    dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})

    def preprocess(example):
        inputs = tokenizer(example["text"], max_length=1024, truncation=True, padding="max_length")
        targets = tokenizer(example["summary"], max_length=128, truncation=True, padding="max_length")
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["text", "summary"])

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=num_train_epochs,
        fp16=False,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=abstractive_model)

    trainer = Trainer(
        model=abstractive_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    abstractive_model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    global abstractive_summarizer
    abstractive_summarizer = pipeline("summarization", model=abstractive_model, tokenizer=tokenizer)

    return jsonify({"message": "Training complete and model reloaded for summarization."})


if __name__ == '__main__':
    app.run(debug=True)
