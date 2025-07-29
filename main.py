# import os
# import json
# import random

# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# import numpy as np


# import torch
# import torch.nn as nn
# import  torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset 
  
# class ChatbotModel(nn.Module):

#     def __init__(self, input_size, output_size):
#         super(ChatbotModel, self).__init__()

#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, output_size)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)

#         return x


# class ChatbotAssistant:

#     def __init__(self, intents_path, function_mappings = None):
#         self.model = None
#         self.intents_path = intents_path

#         self.documents = []
#         self.vocabulary = []
#         self.intents = []
#         self.intents_responses = {}

#         self.function_mappings = function_mappings

#         self.X = None
#         self.y = None

#     @staticmethod
#     def tokenize_and_lemmatize(text):
#         lemmatizer = nltk.WordNetLemmatizer()
#         from nltk.tokenize import TreebankWordTokenizer
#         tokenizer = TreebankWordTokenizer()
#         words = tokenizer.tokenize(text)
#         words = [lemmatizer.lemmatize(word.lower()) for word in words]
#         return words

# chatbot=ChatbotAssistant('intents.json')
# print(chatbot.tokenize_and_lemmatize('hello world how are you ,i am programming on python today'))
import os
import json
import random

import nltk
import numpy as np
import hashlib
import pickle
import requests
from bs4 import BeautifulSoup
import PyPDF2
import openpyxl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 

import re
from spellchecker import SpellChecker
from difflib import get_close_matches
  
class ChatbotModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.function_mappings = function_mappings

        self.X = None
        self.y = None

    @staticmethod
    def normalize_text(text):
        # Lowercase, remove extra spaces, replace hyphens with spaces
        text = text.lower()
        text = re.sub(r'[-_]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def canonicalize(text):
        # Lowercase, remove spaces and hyphens/underscores for canonical comparison
        text = text.lower()
        text = re.sub(r'[-_\s]', '', text)
        return text

    @staticmethod
    def correct_spelling_phrase(text):
        try:
            from textblob import TextBlob
            corrected = str(TextBlob(text).correct())
            return corrected
        except Exception:
            # Fallback to pyspellchecker word-by-word
            spell = SpellChecker()
            words = text.split()
            corrected = [spell.correction(word) for word in words]
            return ' '.join(corrected)

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        from nltk.tokenize import TreebankWordTokenizer
        tokenizer = TreebankWordTokenizer()
        # Normalize text before tokenizing
        text = ChatbotAssistant.normalize_text(text)
        # Spelling correction for the whole phrase
        text = ChatbotAssistant.correct_spelling_phrase(text)
        words = tokenizer.tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words]
        return words

    def canonicalize_pattern_list(self, patterns):
        # Return a set of canonicalized patterns
        return set(self.canonicalize(p) for p in patterns)

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        url = 'https://zemenbazaar.com/en'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- Scrape categories from homepage ---
        categories = soup.find_all('a', class_='category')  # Adjust selector as needed
        for cat in categories:
            cat_name = cat.get_text(strip=True)
            cat_desc = cat.get('title', cat_name)  # Use title attribute or name
            tag = cat_name.lower().replace(' ', '_')[:30]
            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = [cat_desc]
            pattern_words = self.tokenize_and_lemmatize(cat_name)
            self.vocabulary.extend(pattern_words)
            self.documents.append((pattern_words, tag))

        # --- Scrape products from homepage (or category pages) ---
        products = soup.find_all(class_='product')  # Adjust selector as needed
        for prod in products:
            prod_name = prod.find(class_='product-title').get_text(strip=True) if prod.find(class_='product-title') else None
            prod_desc = prod.find(class_='product-description').get_text(strip=True) if prod.find(class_='product-description') else prod_name
            if prod_name:
                tag = prod_name.lower().replace(' ', '_')[:30]
                if tag not in self.intents:
                    self.intents.append(tag)
                    self.intents_responses[tag] = [prod_desc]
                pattern_words = self.tokenize_and_lemmatize(prod_name)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, tag))

        # --- Scrape contact/help info ---
        contact = soup.find(string=lambda t: 'contact' in t.lower())
        if contact:
            tag = 'contact_info'
            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = [contact.strip()]
            pattern_words = self.tokenize_and_lemmatize('contact')
            self.vocabulary.extend(pattern_words)
            self.documents.append((pattern_words, tag))

        # --- Scrape help info ---
        help_info = soup.find(string=lambda t: 'help' in t.lower())
        if help_info:
            tag = 'help_info'
            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = [help_info.strip()]
            pattern_words = self.tokenize_and_lemmatize('help')
            self.vocabulary.extend(pattern_words)
            self.documents.append((pattern_words, tag))

        self.vocabulary = sorted(set(self.vocabulary))

    def load_intents_from_file(self):
        # Load intents from the intents.json file and populate documents and vocabulary
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.intents = []
        self.intents_responses = {}
        self.documents = []
        self.vocabulary = []
        for intent in data['intents']:
            tag = intent['tag']
            self.intents.append(tag)
            self.intents_responses[tag] = intent['responses']
            for pattern in intent['patterns']:
                pattern_words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, tag))
        self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents)) 

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss
            
            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")

    def save_pickle(self, pkl_path):
        data = {
            'model_state': self.model.state_dict(),
            'vocabulary': self.vocabulary,
            'intents': self.intents,
            'intents_responses': self.intents_responses,
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(self, pkl_path, input_size=None, output_size=None):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.vocabulary = data['vocabulary']
        self.intents = data['intents']
        self.intents_responses = data['intents_responses']
        if input_size is None:
            input_size = len(self.vocabulary)
        if output_size is None:
            output_size = len(self.intents)
        self.model = ChatbotModel(input_size, output_size)
        self.model.load_state_dict(data['model_state'])

    def fuzzy_match_intent(self, input_words):
        # Try to find the closest matching intent pattern using fuzzy matching
        all_patterns = []
        for tag in self.intents:
            for pattern in self.intents_responses.get(tag, []):
                all_patterns.append((pattern, tag))
        input_text = ' '.join(input_words)
        matches = get_close_matches(input_text, [p[0].lower() for p in all_patterns], n=1, cutoff=0.8)
        if matches:
            for pattern, tag in all_patterns:
                if pattern.lower() == matches[0]:
                    return tag
        return None

    def fuzzy_phrase_match_intent(self, input_phrase, threshold=0.6):
        # Canonicalize input phrase
        input_canon = self.canonicalize(input_phrase)
        best_match = None
        best_tag = None
        best_score = 0
        for intent in self.intents:
            patterns = self.get_patterns_for_intent(intent)
            for pattern in patterns:
                pattern_canon = self.canonicalize(pattern)
                # Use difflib SequenceMatcher for similarity
                from difflib import SequenceMatcher
                score = SequenceMatcher(None, input_canon, pattern_canon).ratio()
                if score > best_score and score >= threshold:
                    best_score = score
                    best_tag = intent
                    best_match = pattern
        return best_tag

    def process_message(self, input_message, confidence_threshold=0.5):
        # Check for product match first (live web data)
        products = fetch_products()
        if products:
            product = find_product_by_name(products, input_message)
            if product:
                return format_product_response(product)
        # Correct spelling for the whole phrase before tokenization
        input_message = self.correct_spelling_phrase(self.normalize_text(input_message))
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)
            probabilities = torch.softmax(predictions, dim=1)
            max_prob, predicted_class_index = torch.max(probabilities, dim=1)
            predicted_intent = self.intents[predicted_class_index.item()]

        # Canonicalize user input for direct pattern match
        input_canon = self.canonicalize(' '.join(words))
        for intent in self.intents:
            canon_patterns = self.canonicalize_pattern_list(self.get_patterns_for_intent(intent))
            if input_canon in canon_patterns:
                if self.intents_responses[intent]:
                    return random.choice(self.intents_responses[intent])

        # Fuzzy phrase match before fallback
        fuzzy_phrase_tag = self.fuzzy_phrase_match_intent(input_message)
        if fuzzy_phrase_tag and self.intents_responses.get(fuzzy_phrase_tag):
            return random.choice(self.intents_responses[fuzzy_phrase_tag])

        if max_prob.item() < confidence_threshold:
            # Try fuzzy matching as a fallback (word-level)
            fuzzy_tag = self.fuzzy_match_intent(words)
            if fuzzy_tag and self.intents_responses.get(fuzzy_tag):
                return random.choice(self.intents_responses[fuzzy_tag])
            # Fallback response
            if "fallback" in self.intents_responses:
                return random.choice(self.intents_responses["fallback"])
            else:
                return "I'm sorry, I don't understand that question yet."

        if self.function_mappings and predicted_intent in self.function_mappings:
            self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None

    def get_patterns_for_intent(self, intent):
        # Helper to get all patterns for an intent from intents.json
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data['intents']:
            if item['tag'] == intent:
                return item['patterns']
        return []


def file_hash(filepath):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_stocks():
    stocks = ['APPL', 'META', 'NVDA', 'GS', 'MSFT']

    print(random.sample(stocks, 3))


def extract_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def generate_intents_from_text(text):
    # This is a simple heuristic. For a real system, use NLP to extract Q&A pairs or sections.
    # Here, we split by lines and look for feature-like statements.
    import re
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    intents = []
    for i, line in enumerate(lines):
        # Use lines that look like features or section headers
        if (re.match(r"^[0-9]+\. ", line) or
            re.match(r"^[A-Z][A-Za-z\s]+:$", line) or
            (len(line.split()) < 10 and line.isupper())):
            tag = re.sub(r'[^a-zA-Z0-9]', '_', line.lower())[:30]
            # Patterns: user might ask about this feature
            patterns = [
                f"Tell me about {line}",
                f"What is {line}?",
                f"How does {line} work?",
                f"Explain {line}",
                f"Can you help me with {line}?"
            ]
            # Response: use the next few lines as a response
            response_lines = []
            for j in range(i+1, min(i+4, len(lines))):
                if not (re.match(r"^[0-9]+\. ", lines[j]) or re.match(r"^[A-Z][A-Za-z\s]+:$", lines[j])):
                    response_lines.append(lines[j])
                else:
                    break
            response = " ".join(response_lines) if response_lines else f"Here is some information about {line}."
            intents.append({
                "tag": tag,
                "patterns": patterns,
                "responses": [response]
            })
    return intents

def merge_intents(existing, new):
    # Avoid duplicate tags, merge patterns/responses if tag exists
    tags = {intent['tag']: intent for intent in existing}
    for intent in new:
        if intent['tag'] in tags:
            tags[intent['tag']]['patterns'].extend(
                p for p in intent['patterns'] if p not in tags[intent['tag']]['patterns'])
            tags[intent['tag']]['responses'].extend(
                r for r in intent['responses'] if r not in tags[intent['tag']]['responses'])
        else:
            tags[intent['tag']] = intent
    return list(tags.values())

def fetch_products():
    url = "http://147.93.94.157:3000/api/v1/products"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching products: {e}")
        return []

def find_product_by_name(products, query):
    from difflib import get_close_matches
    names = [p.get('name', '') for p in products]
    matches = get_close_matches(query, names, n=1, cutoff=0.7)
    if matches:
        for p in products:
            if p.get('name', '') == matches[0]:
                return p
    # Try substring match if no close match
    for p in products:
        if query.lower() in p.get('name', '').lower():
            return p
    return None

def format_product_response(product):
    name = product.get('name', '')
    price = product.get('price', '')
    description = product.get('description', '')
    product_id = str(product.get('id', '')).strip()
    link = f"https://zemenbazaar.com/en/products/{product_id}" if product_id else ''
    return f"Name: {name}\nPrice: {price}\nLink: {link}\nDescription: {description}"

if __name__ == '__main__':
    assistant = ChatbotAssistant('intents.json', function_mappings = {'stocks': get_stocks})
    assistant.load_intents_from_file()  # Use intents.json and PDF only

    model_exists = os.path.exists('chatbot_model.pkl')
    intents_hash_path = 'intents.hash'
    current_hash = file_hash('intents.json')
    previous_hash = None
    if os.path.exists(intents_hash_path):
        with open(intents_hash_path, 'r') as f:
            previous_hash = f.read().strip()

    needs_retrain = (not model_exists) or (previous_hash != current_hash)
    if needs_retrain:
        assistant.prepare_data()
        assistant.train_model(batch_size=8, lr=0.001, epochs=100)
        assistant.save_pickle('chatbot_model.pkl')
        with open(intents_hash_path, 'w') as f:
            f.write(current_hash)
        assistant.load_pickle('chatbot_model.pkl')
    else:
        assistant.load_pickle('chatbot_model.pkl')

    # PDF extraction and intent merging utility
    pdf_path = 'How the Zemen Bazaar Works.pdf'
    if os.path.exists(pdf_path):
        pdf_text = extract_pdf_text(pdf_path)
        pdf_intents = generate_intents_from_text(pdf_text)
        with open('intents.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        merged_intents = merge_intents(data['intents'], pdf_intents)
        data['intents'] = merged_intents
        with open('intents.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Merged {len(pdf_intents)} PDF-based intents into intents.json.")

    while True:
        message = input('Enter your message:')

        if message == '/quit':
            break

        print(assistant.process_message(message))