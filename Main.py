import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from transformers import BertForTokenClassification, BertTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import MarianMTModel, MarianTokenizer
from newspaper import Article
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import trafilatura
import feedparser
import sqlite3
import folium
from opencage.geocoder import OpenCageGeocode
from langdetect import detect


# Load the classification model and tokenizer
model_path = 'D:\flask_app\Trained_Model\lightning_strike_classifier'
try:
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path, from_tf=False, local_files_only=True)
    print("Classification model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading classification model or tokenizer: {e}")

# Initialize NER pipeline with BERT model
model_name_ner = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer_ner = BertTokenizer.from_pretrained(model_name_ner)
model_ner = BertForTokenClassification.from_pretrained(model_name_ner)
ner_model = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner, grouped_entities=True)
print("NER model loaded successfully.")

# Initialize BART model for summarization
model_name_summarization = "facebook/bart-large-cnn"
tokenizer_summarization = BartTokenizer.from_pretrained(model_name_summarization)
model_summarization = BartForConditionalGeneration.from_pretrained(model_name_summarization)
print("BART model for summarization loaded successfully.")

# Initialize translation model
translation_model_name = "Helsinki-NLP/opus-mt-mul-en"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)
print("Translation model loaded successfully.")

# Your OpenCage API key
API_KEY = 'df53299bc8f54459954b67d2e781f197'

# Function to read RSS feed URLs from a CSV file
def read_rss_urls_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        rss_urls = df['rss_url'].tolist()  # Assuming the CSV file has a column named 'rss_url'
        return rss_urls
    except Exception as e:
        print(f"Error reading RSS URLs from CSV: {e}")
        return []

# Function to get latitude and longitude from OpenCage Geocoding API
def get_lat_long(location):
    geocoder = OpenCageGeocode(API_KEY)
    try:
        results = geocoder.geocode(location)
        if results:
            return results[0]['geometry']['lat'], results[0]['geometry']['lng']
        else:
            return None, None
    except Exception as e:
        print(f"Error fetching geocoding data for {location}: {e}")
        return None, None

# Function to predict the label for a given text
def predict_label(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=1).item()
        return predicted_label
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Function to extract locations from a given text using NER
def extract_locations(text):
    try:
        entities = ner_model(text)
        locations = [entity['word'] for entity in entities if entity['entity_group'] == 'LOC']
        return locations
    except Exception as e:
        print(f"Error during NER extraction: {e}")
        return []

# Function to parse RSS feed
def parse_rss_feed(rss_url):
    feed = feedparser.parse(rss_url)
    news_items = []

    for entry in feed.entries:
        title = entry.get("title", "")
        published_date = entry.get("published") or entry.get("updated") or entry.get("pubDate") or ""
        link = entry.get("link", "")

        news_item = {
            "title": title,
            "published_date": published_date,
            "link": link
        }
        news_items.append(news_item)

    return news_items

# Function to format the date to dd/mm/yyyy
def format_date(date_str):
    try:
        date = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z')
        return date.strftime('%d/%m/%Y')
    except ValueError:
        try:
            date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
            return date.strftime('%d/%m/%Y')
        except ValueError:
            return date_str

# Function to extract the main content of a news article using newspaper3k
def extract_main_content(url, num_sentences=5):
    try:
        downloaded = trafilatura.fetch_url(url)
        main_text = trafilatura.extract(downloaded)
        print("Trafilatura:", main_text)

        if not main_text:
            print("Trafilatura failed, falling back to newspaper3k.")
            article = Article(url)
            article.download()
            article.parse()
            main_text = article.text

            if not main_text or len(main_text.split()) < 100:
                print("Newspaper3k failed, falling back to BeautifulSoup.")
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                main_text = ''
                for tag in soup.find_all(['p', 'div', 'article']):
                    text = tag.get_text(strip=True)
                    if text and len(text.split()) > 10:
                        main_text += text + '\n'

        # Extract the first few sentences
        sentences = main_text.split('. ')
        shortened_content = '. '.join(sentences[:num_sentences]) + '.'
        return shortened_content
    except Exception as e:
        print(f"Error occurred during content extraction: {e}")
        return None

# Function to summarize content using BART including the title
def summarize_content(title, content, lang):
    try:
        if not content:
            return "No content to summarize"

        if len(content.split()) < 10:
            return "Insufficient content for summarization"

        # Set the tokenizer language
        tokenizer_summarization.src_lang = lang
        tokenizer_summarization.tgt_lang = lang

        # Combine title and shortened content for summarization
        text_to_summarize = f"Title: {title}\n\nContent: {content}"

        summarizer = pipeline("summarization", model=model_summarization, tokenizer=tokenizer_summarization)
        summary = summarizer(text_to_summarize, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Failed to summarize content"

# Function to detect the language of a text
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Error detecting language: {e}")
        return None

# Function to translate text to English using MarianMT
def translate_to_english(text, src_lang):
    try:
        translated = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translation = translation_model.generate(**translated)
        translated_text = translation_tokenizer.batch_decode(translation, skip_special_tokens=True)
        return translated_text[0]
    except Exception as e:
        print(f"Error during translation: {e}")
        return text  # Fallback to the original text if translation fails

# Initialize database connection
def init_db():
    conn = sqlite3.connect('D:\flask_app\Database\news_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS news (
            date TEXT,
            longitude REAL,
            latitude REAL,
            link TEXT,
            locations TEXT,
            PRIMARY KEY (date, longitude, latitude)
        )
    ''')
    conn.commit()
    return conn

# Function to insert data into database
def insert_data(conn, link, date, locations, latitude, longitude):
    cursor = conn.cursor()
    try:
        # Check if the entry already exists
        cursor.execute('''
            SELECT 1 FROM news WHERE date = ? AND longitude = ? AND latitude = ?
        ''', (date, longitude, latitude))
        exists = cursor.fetchone()

        if not exists:
            cursor.execute('''
                INSERT INTO news (link, date, locations, latitude, longitude) VALUES (?, ?, ?, ?, ?)
            ''', (link, date, ', '.join(locations), latitude, longitude))
            conn.commit()
            print("Data inserted successfully.")
        else:
            print("Skipping insertion due to duplicate entry.")
    except Exception as e:
        print(f"Error inserting data: {e}")

# Main execution flow
if __name__ == "__main__":
    # Initialize database
    conn = init_db()

    # Path to the CSV file containing RSS feed URLs
    csv_file_path = 'D:\flask_app\Input\Input_rss.csv'

    # Read RSS feed URLs from the CSV file
    rss_urls = read_rss_urls_from_csv(csv_file_path)

    extracted_info = []

    for rss_url in rss_urls:
        news_items = parse_rss_feed(rss_url)

        for item in news_items:
            title = item["title"]
            url = item["link"]
            published_date = format_date(item["published_date"])
            print("Original title:", title)

            language = detect_language(title)
            if language and language != 'en':
                title = translate_to_english(title, language)
                print("Translated title:", title)
                content = extract_main_content(url, num_sentences=5)
                if content:
                    content = translate_to_english(content, language)
                    print("Translated content:", content)
            else:
                content = extract_main_content(url, num_sentences=5)

            is_lightning_related = predict_label(title)
            print(is_lightning_related)
            if is_lightning_related == 1:
                if content:
                    summarized_content = summarize_content(title, content, "en")

                    if summarized_content == "Failed to summarize content":
                        print("Failed to summarize content. Extracting locations from title.")
                        locations = extract_locations(title)
                    else:
                        locations = extract_locations(summarized_content)

                    if locations:
                        latitude, longitude = get_lat_long(locations[0])
                    else:
                        latitude, longitude = None, None

                    print("Published Date:", published_date)
                    print("Link:", url)
                    if summarized_content != "Failed to summarize content":
                        print("News Content:", summarized_content)
                    if locations:
                        print(f"Locations mentioned: {locations}")
                    else:
                        print("No locations extracted.")

                    extracted_info.append({
                        "link": url,
                        "date": published_date,
                        "locations": locations,
                        "latitude": latitude,
                        "longitude": longitude
                    })

                    insert_data(conn, url, published_date, locations, latitude, longitude)

                else:
                    print("Failed to extract content. Extracting locations from title.")
                    locations = extract_locations(title)
                    if locations:
                        latitude, longitude = get_lat_long(locations[0])
                    else:
                        latitude, longitude = None, None

                    print("Published Date:", published_date)
                    print("Link:", url)
                    if locations:
                        print(f"Locations mentioned in title: {locations}")
                    else:
                        print("No locations extracted from title.")

                    extracted_info.append({
                        "link": url,
                        "date": published_date,
                        "locations": locations,
                        "latitude": latitude,
                        "longitude": longitude
                    })

                    insert_data(conn, url, published_date, locations, latitude, longitude)

            else:
                print("The news is not related to lightning.")

            print("\n")

    print("Extracted Information:")
    for info in extracted_info:
        print(info)

    conn.close()
