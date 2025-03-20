import pandas as pd
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nlpaug.augmenter.word as naw

# Load your dataset
df = pd.read_csv('D:\flask_app\Dataset\Train_data.csv')

# Ensure the dataset has 'text' and 'label' columns
text_column = 'Text'
label_column = 'Label'

# Convert labels to integers if they are not already
df[label_column] = df[label_column].astype(int)

# Ensure 'Text' column is of string type and not nested
df[text_column] = df[text_column].astype(str)

# Data Augmentation
aug = naw.SynonymAug(aug_src='wordnet')
augmented_texts = [aug.augment(text) for text in df[text_column]]
augmented_df = pd.DataFrame({text_column: augmented_texts, label_column: df[label_column]})
df = pd.concat([df, augmented_df])

# Ensure augmented texts are of string type
df[text_column] = df[text_column].astype(str)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples[text_column], padding='max_length', truncation=True, max_length=512)

# Hyperparameter tuning and cross-validation
skf = StratifiedKFold(n_splits=5)
for train_index, val_index in skf.split(df[text_column], df[label_column]):
    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]

    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Tokenize the dataset
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    train_dataset = train_dataset.rename_column(label_column, "labels")
    val_dataset = val_dataset.rename_column(label_column, "labels")

    # Set the format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Load the model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        save_total_limit=2,
    )

    # Define compute_metrics function
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)]
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained('D:\flask_app\Trained_Model\lightning_strike_classifier2')
    tokenizer.save_pretrained('D:\flask_app\Trained_Model\lightning_strike_classifier2')
