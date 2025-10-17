#!/usr/bin/env python3
"""
Flask Backend untuk SportExtract - Website NER Berita Olahraga
Mengintegrasikan model IndoBERT terbaik untuk ekstraksi entitas
Mendukung scraping dari DetikSport dan Kompas Sport
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict

# Add parent directory to path untuk import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train_model'))

class IndoBERTNERModelMultiLabel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        try:
            self.bert = AutoModel.from_pretrained(model_name, use_safetensors=True)
        except:
            try:
                self.bert = AutoModel.from_pretrained(model_name, use_safetensors=False)
            except:
                self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1, labels.size(-1))[active_loss]
            loss = loss_fct(active_logits, active_labels)
        
        return {'loss': loss, 'logits': logits}

class NERInference:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.label2id = None
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load model dari checkpoint"""
        print(f"🔄 Loading model from: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            config = checkpoint.get('config', {})
            model_name = config.get('model_name', 'indobenchmark/indobert-base-p1')
            self.id2label = checkpoint.get('id2label', {})
            self.label2id = {v: k for k, v in self.id2label.items()}
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.model = IndoBERTNERModelMultiLabel(model_name, len(self.id2label))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully!")
            print(f"   Available labels: {list(self.id2label.values())}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_entities(self, text, threshold=0.5):
        """Prediksi entities dari teks"""
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            padding='max_length'
        )
        
        input_ids = torch.tensor([encoding['input_ids']], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([encoding['attention_mask']], dtype=torch.long).to(self.device)
        offset_mapping = encoding['offset_mapping']
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            predictions = torch.sigmoid(logits) > threshold
        
        # Extract entities
        entities = self.extract_entities(text, predictions[0], offset_mapping, attention_mask[0])
        
        return entities
    
    def extract_entities(self, text, predictions, offset_mapping, attention_mask):
        """Extract entities dari predictions dengan subword merging"""
        entities = []
        
        for token_idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == token_end or attention_mask[token_idx] == 0:
                continue
            
            # Get active labels for this token
            active_labels = []
            for label_id, is_active in enumerate(predictions[token_idx]):
                if is_active:
                    label_name = self.id2label.get(label_id, f"LABEL_{label_id}")
                    active_labels.append(label_name)
            
            if active_labels:
                token_text = text[token_start:token_end]
                entities.append({
                    'text': token_text,
                    'start': token_start,
                    'end': token_end,
                    'labels': active_labels
                })
        
        # Merge consecutive subwords with same labels
        merged_entities = self._merge_subword_entities(entities, text)
        return merged_entities
    
    def _merge_subword_entities(self, entities, text):
        """Merge consecutive subword tokens dengan label yang sama"""
        if not entities:
            return []
        
        merged = []
        current_entity = None
        
        for entity in entities:
            if current_entity is None:
                current_entity = entity.copy()
            else:
                # Check if we can merge with previous entity
                gap = text[current_entity['end']:entity['start']]
                same_labels = set(current_entity['labels']) == set(entity['labels'])
                small_gap = len(gap.strip()) <= 1  # Allow small gaps like spaces
                
                if same_labels and small_gap:
                    # Merge entities
                    current_entity['end'] = entity['end']
                    current_entity['text'] = text[current_entity['start']:current_entity['end']]
                else:
                    # Save current entity and start new one
                    merged.append(current_entity)
                    current_entity = entity.copy()
        
        # Add the last entity
        if current_entity:
            merged.append(current_entity)
        
        return merged
    
    def format_entities_for_5w1h(self, entities):
        """Format entities untuk 5W1H display"""
        # Mapping entity types ke 5W1H - LENGKAP
        w5h1_mapping = {
            # WHO (Siapa) - Orang, Tim, Organisasi
            'ATLET': 'who',           # Pemain sepak bola
            'TIM': 'who',             # Tim sepak bola
            'ORGANISASI': 'who',      # Organisasi sepak bola
            'KEWARGANEGARAAN': 'who', # Negara asal
            'POSISI': 'who',          # Posisi pemain
            'UMUR': 'who',            # Usia pemain
            
            # WHAT (Apa) - Aksi, Hasil, Statistik
            'AKSI': 'what',           # Aksi dalam pertandingan
            'PENGHARGAAN': 'what',    # Penghargaan/prestasi
            'STATISTIK': 'what',      # Data statistik (AKAN DIGABUNG JADI KALIMAT)
            'SKOR': 'what',           # Hasil pertandingan
            
            # WHEN (Kapan) - Waktu
            'TANGGAL': 'when',        # Waktu pertandingan
            
            # WHERE (Dimana) - Tempat
            'STADION': 'where',       # Tempat pertandingan
            'KEJUARAAN': 'where',     # Kompetisi/turnamen
            
            # WHY (Mengapa) - Alasan
            'ALASAN_PERISTIWA': 'why', # Konteks peristiwa
            
            # HOW (Bagaimana) - Cara/Metode
            # Note: HOW biasanya diambil dari AKSI yang lebih spesifik
            # atau kombinasi dari beberapa entitas
        }
        
        # Label yang harus digabungkan menjadi satu kesatuan
        labels_to_merge = ['ATLET', 'ORGANISASI', 'KEWARGANEGARAAN', 'POSISI', 'UMUR', 'TIM', 'PENGHARGAAN', 'SKOR']
        
        # Label yang akan diproses sebagai kalimat lengkap
        labels_for_sentences = ['STATISTIK', 'AKSI', 'ALASAN_PERISTIWA', 'STADION', 'KEJUARAAN', 'TANGGAL']
        
        w5h1_results = {
            'who': [],
            'what': [],
            'when': [],
            'where': [],
            'why': [],
            'how': []
        }
        
        # Group entities by label untuk merge
        entities_by_label = {}
        for entity in entities:
            for label in entity['labels']:
                if label not in entities_by_label:
                    entities_by_label[label] = []
                entities_by_label[label].append(entity)
        
        # Process entities yang perlu di-merge
        for label in labels_to_merge:
            if label not in entities_by_label or label not in w5h1_mapping:
                continue
            
            category = w5h1_mapping[label]
            label_entities = entities_by_label[label]
            
            # Gabungkan entities dengan label yang sama (returns list now)
            merged_texts = self._combine_entities_smart(label_entities)
            
            # Tambahkan semua hasil grup ke category
            for merged_text in merged_texts:
                if merged_text and len(merged_text) > 2:
                    w5h1_results[category].append(merged_text)
        
        # Generate HOW sebagai kalimat lengkap dari AKSI
        w5h1_results['how'] = self._generate_how_sentences(entities)
        
        # Generate WHY sebagai kalimat lengkap dari ALASAN_PERISTIWA
        w5h1_results['why'] = self._generate_why_sentences(entities)
        
        # Generate STATISTIK sebagai kalimat lengkap dan TAMBAHKAN ke WHAT
        statistik_sentences = self._generate_statistik_sentences(entities)
        w5h1_results['what'].extend(statistik_sentences)
        
        # Generate WHERE sebagai kalimat lengkap dari STADION dan KEJUARAAN
        where_sentences = self._generate_where_sentences(entities)
        w5h1_results['where'].extend(where_sentences)
        
        # Generate WHEN sebagai kalimat lengkap dari TANGGAL
        when_sentences = self._generate_when_sentences(entities)
        w5h1_results['when'].extend(when_sentences)
        
        # Remove duplicates untuk yang sudah di-merge
        for category in w5h1_results:
            w5h1_results[category] = list(dict.fromkeys(w5h1_results[category]))  # Preserve order
        
        return w5h1_results
    
    def _combine_entities_smart(self, entity_list, max_gap=50):
        """Combine entities dengan deduplikasi otomatis dan grouping berdasarkan kedekatan"""
        if not entity_list:
            return []
        
        # Sort berdasarkan posisi start
        entity_list.sort(key=lambda x: x['start'])
        
        # Group entities yang berdekatan (dalam jarak max_gap karakter)
        groups = []
        current_group = [entity_list[0]]
        
        for i in range(1, len(entity_list)):
            prev_entity = current_group[-1]
            curr_entity = entity_list[i]
            
            # Hitung jarak antara entity sebelumnya dan sekarang
            gap = curr_entity['start'] - prev_entity['end']
            
            if gap <= max_gap:
                # Masih dalam grup yang sama
                current_group.append(curr_entity)
            else:
                # Mulai grup baru
                groups.append(current_group)
                current_group = [curr_entity]
        
        # Tambahkan grup terakhir
        if current_group:
            groups.append(current_group)
        
        # Combine setiap grup menjadi string
        results = []
        for group in groups:
            words_seen = set()
            result_words = []
            
            for entity in group:
                entity_text = entity['text'].strip()
                
                # Skip jika terlalu pendek atau hanya tanda baca
                if len(entity_text) <= 1 or entity_text in ['.', ',', ';', ':', '!', '?']:
                    continue
                
                # Split menjadi kata-kata untuk cek duplikasi dalam grup ini
                words = entity_text.split()
                
                for word in words:
                    word_lower = word.lower()
                    # Hanya tambahkan jika belum pernah muncul dalam grup ini
                    if word_lower not in words_seen and len(word) > 1:
                        words_seen.add(word_lower)
                        result_words.append(word)
            
            # Gabungkan kembali
            combined = " ".join(result_words).strip()
            
            # Tambahkan jika memenuhi kriteria minimal
            if len(combined) > 5:
                results.append(combined)
        
        return results
    
    def _generate_statistik_sentences(self, entities):
        """Generate STATISTIK sebagai kalimat lengkap (untuk WHAT)"""
        statistik_entities = [e for e in entities if 'STATISTIK' in e['labels']]
        
        if not statistik_entities:
            return []
        
        # Returns list of strings now
        return self._combine_entities_smart(statistik_entities)
    
    def _generate_where_sentences(self, entities):
        """Generate WHERE sebagai kalimat lengkap dari STADION dan KEJUARAAN (terpisah)"""
        results = []
        
        # Ambil semua STADION entities dan gabungkan
        stadion_entities = [e for e in entities if 'STADION' in e['labels']]
        if stadion_entities:
            combined_stadion = self._combine_entities_smart(stadion_entities)
            results.extend(combined_stadion)
        
        # Ambil semua KEJUARAAN entities dan gabungkan
        kejuaraan_entities = [e for e in entities if 'KEJUARAAN' in e['labels']]
        if kejuaraan_entities:
            combined_kejuaraan = self._combine_entities_smart(kejuaraan_entities)
            results.extend(combined_kejuaraan)
        
        return results
    
    def _generate_when_sentences(self, entities):
        """Generate WHEN sebagai kalimat lengkap dari TANGGAL"""
        when_entities = [e for e in entities if 'TANGGAL' in e['labels']]
        
        if not when_entities:
            return []
        
        # Returns list of strings now
        return self._combine_entities_smart(when_entities)
    
    def _generate_how_sentences(self, entities):
        """Generate HOW sebagai kalimat lengkap dari AKSI"""
        aksi_entities = [e for e in entities if 'AKSI' in e['labels']]
        
        if not aksi_entities:
            return []
        
        # Returns list of strings now
        return self._combine_entities_smart(aksi_entities)
    
    def _generate_why_sentences(self, entities):
        """Generate WHY sebagai kalimat lengkap dari ALASAN_PERISTIWA"""
        alasan_entities = [e for e in entities if 'ALASAN_PERISTIWA' in e['labels']]
        
        if not alasan_entities:
            return []
        
        # Returns list of strings now
        return self._combine_entities_smart(alasan_entities)

class WebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'id-ID,id;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def scrape_detiksport(self, url):
        """Scrape artikel dari DetikSport"""
        try:
            print(f"🔄 Scraping DetikSport: {url}")
            
            # Parse URL untuk validasi
            parsed_url = urlparse(url)
            if 'detik.com' not in parsed_url.netloc:
                raise ValueError("URL bukan dari DetikSport")
            
            # Request halaman
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_selectors = [
                'h1.detail__title',
                'h1.title',
                'h1',
                '.detail__title',
                '.title'
            ]
            title = ""
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break
            
            # Extract content
            content_selectors = [
                '.detail__body-text',
                '.article-content',
                '.content',
                '.detail__body',
                '.read__content'
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove unwanted elements
                    for unwanted in content_elem.select('script, style, .ads, .advertisement, .social-share'):
                        unwanted.decompose()
                    
                    # Extract text from paragraphs
                    paragraphs = content_elem.find_all(['p', 'div'])
                    content_parts = []
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if text and len(text) > 10:  # Filter short text
                            if 'ADVERTISEMENT' in text or 'SCROLL TO CONTINUE WITH CONTENT' in text:
                                continue
                            content_parts.append(text)
                    
                    content = '\n\n'.join(content_parts)
                    break
            
            # Extract metadata
            author = ""
            author_elem = soup.select_one('.detail__author, .author, .penulis')
            if author_elem:
                author = author_elem.get_text(strip=True)
            
            date = ""
            date_elem = soup.select_one('.detail__date, .date, .tanggal')
            if date_elem:
                date = date_elem.get_text(strip=True)
            
            # Extract tags/category
            tags = []
            tag_elements = soup.select('.detail__tags a, .tags a, .category a')
            for tag_elem in tag_elements:
                tag_text = tag_elem.get_text(strip=True)
                if tag_text:
                    tags.append(tag_text)
            
            # Clean content
            if content:
                content = self.clean_text(content)
            
            result = {
                'title': title,
                'content': content,
                'author': author,
                'date': date,
                'tags': tags,
                'url': url,
                'source': 'DetikSport',
                'word_count': len(content.split()) if content else 0,
                'char_count': len(content) if content else 0
            }
            
            print(f"✅ Successfully scraped: {title[:50]}...")
            return result
            
        except Exception as e:
            print(f"❌ Error scraping DetikSport: {e}")
            raise Exception(f"Gagal mengambil konten dari DetikSport: {str(e)}")
    
    def scrape_kompas(self, url):
        """Scrape artikel dari Kompas Sport (bola.kompas.com dan kompas.com)"""
        try:
            print(f"🔄 Scraping Kompas Sport: {url}")
            
            # Parse URL untuk validasi
            parsed_url = urlparse(url)
            if 'kompas.com' not in parsed_url.netloc:
                raise ValueError("URL bukan dari Kompas")
            
            # Request halaman
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_selectors = [
                'h1.read__title',
                'h1.title',
                'h1',
                '.read__title',
                '.article__title'
            ]
            title = ""
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break
            
            # Extract content - Kompas menggunakan struktur khusus
            content = ""
            
            # Try read__content (untuk artikel olahraga)
            content_elem = soup.select_one('.read__content')
            if content_elem:
                # Remove unwanted elements
                for unwanted in content_elem.select('script, style, .ads, .advertisement, .baca, strong'):
                    unwanted.decompose()
                
                # Extract text from paragraphs
                paragraphs = content_elem.find_all('p')
                content_parts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:  # Filter short text
                        # Skip unwanted patterns
                        if any(skip in text for skip in ['Baca juga:', 'KOMPAS.com', 'Dapatkan update', 'Simak breaking news']):
                            continue
                        content_parts.append(text)
                
                content = '\n\n'.join(content_parts)
            
            # Extract metadata
            author = ""
            author_selectors = [
                '.read__author__name',
                '.author__name', 
                '.author',
                '.penulis'
            ]
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem:
                    author = author_elem.get_text(strip=True)
                    break
            
            date = ""
            date_selectors = [
                '.read__time',
                '.read__date',
                '.article__date',
                '.date'
            ]
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    date = date_elem.get_text(strip=True)
                    break
            
            # Extract tags/category
            tags = []
            tag_elements = soup.select('.tag__article__item a, .tag a, .article__tag a')
            for tag_elem in tag_elements:
                tag_text = tag_elem.get_text(strip=True)
                if tag_text:
                    tags.append(tag_text)
            
            # Clean content
            if content:
                content = self.clean_text(content)
            
            result = {
                'title': title,
                'content': content,
                'author': author,
                'date': date,
                'tags': tags,
                'url': url,
                'source': 'Kompas Sport',
                'word_count': len(content.split()) if content else 0,
                'char_count': len(content) if content else 0
            }
            
            print(f"✅ Successfully scraped: {title[:50]}...")
            return result
            
        except Exception as e:
            print(f"❌ Error scraping Kompas Sport: {e}")
            raise Exception(f"Gagal mengambil konten dari Kompas Sport: {str(e)}")
    
    def clean_text(self, text):
        text = re.sub(r'ADVERTISEMENT', '', text, flags=re.IGNORECASE)
        text = re.sub(r'SCROLL TO CONTINUE WITH CONTENT', '', text, flags=re.IGNORECASE)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\/]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def scrape_url(self, url):
        """Main scraping function - determines which scraper to use"""
        parsed_url = urlparse(url)
        
        if 'detik.com' in parsed_url.netloc:
            return self.scrape_detiksport(url)
        elif 'kompas.com' in parsed_url.netloc:
            return self.scrape_kompas(url)
        else:
            raise ValueError("Website tidak didukung. Saat ini hanya mendukung DetikSport dan Kompas Sport.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model instance
ner_model = None
web_scraper = None

def initialize_model():
    """Initialize model dan web scraper saat startup"""
    global ner_model, web_scraper
    try:
        # Try different possible paths
        possible_paths = [
            '../train_model/outputs_indobert_multilabel/best_model.pt',
            './train_model/outputs_indobert_multilabel/best_model.pt',
            os.path.join(os.path.dirname(__file__), '..', 'train_model', 'outputs_indobert_multilabel', 'best_model.pt')
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            raise FileNotFoundError(f"Model not found in any of these paths: {possible_paths}")
        
        print(f"🔄 Loading model from: {model_path}")
        ner_model = NERInference(model_path, device='cpu')
        
        # Initialize web scraper
        web_scraper = WebScraper()
        
        print("✅ Model and web scraper initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False

@app.route('/')
def index():
    """Serve the main website"""
    return render_template('website.html')

@app.route('/api/extract', methods=['POST'])
def extract_entities():
    """API endpoint untuk ekstraksi entitas"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text input is required'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        if not ner_model:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Check if input is URL
        scraped_data = None
        if text.startswith('http://') or text.startswith('https://'):
            if not web_scraper:
                return jsonify({'error': 'Web scraper not initialized'}), 500
            
            try:
                print(f"🔄 Detected URL, starting scraping: {text}")
                scraped_data = web_scraper.scrape_url(text)
                text = scraped_data['content']  # Use scraped content for NER
                
                if not text:
                    return jsonify({'error': 'Tidak dapat mengambil konten dari URL tersebut'}), 400
                    
            except Exception as scrape_error:
                print(f"❌ Scraping error: {scrape_error}")
                return jsonify({'error': f'Gagal mengambil konten dari URL: {str(scrape_error)}'}), 400
        
        # Extract entities
        entities = ner_model.predict_entities(text)
        
        # Format untuk 5W1H
        w5h1_results = ner_model.format_entities_for_5w1h(entities)
        
        # Prepare response
        response = {
            'success': True,
            'entities': entities,
            'w5h1': w5h1_results,
            'stats': {
                'total_entities': len(entities),
                'text_length': len(text),
                'word_count': len(text.split())
            }
        }
        
        # Add scraped data if available
        if scraped_data:
            response['scraped_data'] = scraped_data
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error in extract_entities: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': ner_model is not None,
        'scraper_loaded': web_scraper is not None,
        'supported_sources': ['DetikSport', 'Kompas Sport'],
        'available_labels': list(ner_model.id2label.values()) if ner_model else []
    })

if __name__ == '__main__':
    print("🚀 Starting SportExtract Flask Server...")
    print("📰 Supported Sources: DetikSport, Kompas Sport (bola.kompas.com & kompas.com)")
    
    # Initialize model
    if not initialize_model():
        print("❌ Failed to initialize model. Exiting...")
        sys.exit(1)
    
    # Create templates directory if not exists
    os.makedirs('templates', exist_ok=True)
    
    # Copy website.html to templates
    import shutil
    if os.path.exists('website.html'):
        shutil.copy('website.html', 'templates/website.html')
        print("✅ Website template copied to templates/")
    
    print("🌐 Server starting on http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)