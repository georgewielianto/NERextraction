import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict
from collections import defaultdict

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
        
        # Print detailed log
        self.print_entity_log(entities)
        
        return entities
    
    def print_entity_log(self, entities):
        """Print detailed log of extracted entities by label"""
        print("\n" + "="*80)
        print("📊 HASIL EKSTRAKSI ENTITAS")
        print("="*80)
        
        # Group entities by label
        entities_by_label = defaultdict(list)
        for entity in entities:
            for label in entity['labels']:
                entities_by_label[label].append(entity['text'])
        
        # Sort labels alphabetically
        sorted_labels = sorted(entities_by_label.keys())
        
        # Print total
        print(f"\n🔢 TOTAL ENTITIES: {len(entities)}")
        print(f"🏷️  TOTAL LABEL TYPES: {len(entities_by_label)}")
        print("\n" + "-"*80)
        
        # Print by label
        for label in sorted_labels:
            texts = entities_by_label[label]
            print(f"\n📌 {label}: ({len(texts)} entitas)")
            for i, text in enumerate(texts, 1):
                print(f"   {i}. {text}")
        
        print("\n" + "="*80)
        print(f"✅ Ekstraksi selesai!\n")
    
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
            
            # WHAT (Apa) - Aksi, Hasil, Statistik, Kompetisi
            'PENGHARGAAN': 'what',    # Penghargaan/prestasi
            'STATISTIK': 'what',      # Data statistik (AKAN DIGABUNG JADI KALIMAT)
            'SKOR': 'what',           # Hasil pertandingan
            
            # WHEN (Kapan) - Waktu
            'TANGGAL': 'when',        # Waktu pertandingan
            
            # WHERE (Dimana) - Tempat
            'STADION': 'where',       # Tempat pertandingan
            'KEJUARAAN': 'where',      # Kompetisi/turnamen - PINDAH KE WHAT

            'AKSI': 'how',           # Aksi dalam pertandingan
            
            # WHY (Mengapa) - Alasan
            'ALASAN_PERISTIWA': 'why', # Konteks peristiwa
            
        
        }
        
        # Label yang harus digabungkan menjadi satu kesatuan
        labels_to_merge = ['ATLET', 'ORGANISASI', 'KEWARGANEGARAAN', 'POSISI', 'UMUR', 'TIM', 'PENGHARGAAN', 'SKOR', 'KEJUARAAN']
        
        # Label yang akan diproses sebagai kalimat lengkap
        labels_for_sentences = ['STATISTIK', 'AKSI', 'ALASAN_PERISTIWA', 'STADION', 'TANGGAL']
        
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
        
        # Generate WHERE sebagai kalimat lengkap dari STADION SAJA (KEJUARAAN SUDAH DI WHAT)
        where_sentences = self._generate_where_sentences(entities)
        w5h1_results['where'].extend(where_sentences)
        
        # Generate WHEN sebagai kalimat lengkap dari TANGGAL
        when_sentences = self._generate_when_sentences(entities)
        w5h1_results['when'].extend(when_sentences)
        
        # Remove duplicates untuk yang sudah di-merge
        for category in w5h1_results:
            w5h1_results[category] = list(dict.fromkeys(w5h1_results[category]))  # Preserve order
        
        # Print 5W1H log
        self.print_5w1h_log(w5h1_results)
        
        return w5h1_results
    
    def print_5w1h_log(self, w5h1_results):
        """Print detailed 5W1H results"""
        print("\n" + "="*80)
        print("📋 HASIL 5W1H")
        print("="*80)
        
        emoji_map = {
            'who': '👤',
            'what': '📌',
            'when': '🕐',
            'where': '📍',
            'why': '❓',
            'how': '🔧'
        }
        
        for category in ['who', 'what', 'when', 'where', 'why', 'how']:
            items = w5h1_results[category]
            emoji = emoji_map.get(category, '•')
            
            print(f"\n{emoji} {category.upper()}: ({len(items)} items)")
            if items:
                for i, item in enumerate(items, 1):
                    print(f"   {i}. {item}")
            else:
                print("   (tidak ada informasi)")
        
        print("\n" + "="*80 + "\n")
    
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
        """Generate WHERE sebagai kalimat lengkap dari STADION SAJA (KEJUARAAN sudah di WHAT)"""
        results = []
        
        # Ambil semua STADION entities dan gabungkan
        stadion_entities = [e for e in entities if 'STADION' in e['labels']]
        if stadion_entities:
            combined_stadion = self._combine_entities_smart(stadion_entities)
            results.extend(combined_stadion)
                
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
    
        self.supported_categories = {
            'detik.com': {
                'allowed_paths': ['/sepakbola', '/basket', '/raket'],
                'sports': ['Sepakbola', 'Basket', 'Raket (Badminton, Tenis)']
                },
            'kompas.com': {
                'allowed_domains': ['bola.kompas.com'],  # Sepakbola
                'allowed_paths': ['/sports', '/badminton'],  # Basket & Badminton
                'sports': ['Sepakbola (bola.kompas.com)', 'Basket (/sports)', 'Badminton (/badminton)']
                },
            'bolasport.com': {
                'allowed_paths': [],  # BolaSport accepts all sports articles
                'sports': ['Semua cabang olahraga']
                }
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


    def scrape_bolasport(self, url):
        """Scrape artikel dari BolaSport.com"""
        try:
            print(f"🔄 Scraping BolaSport: {url}")
            
            # Parse URL untuk validasi
            parsed_url = urlparse(url)
            if 'bolasport.com' not in parsed_url.netloc:
                raise ValueError("URL bukan dari BolaSport")
            
            # Request halaman
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title - BolaSport menggunakan struktur khusus
            title = ""
            title_selectors = [
                'h1.text-title-big',
                'h1.title-large',
                'h1[itemprop="headline"]',
                '.article-header h1',
                'h1'
            ]
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break
            
            # Extract content - BolaSport biasanya menggunakan div dengan class khusus
            content = ""
            content_selectors = [
                '.body-content',
                '.article-content-body',
                '.content-text',
                'div[itemprop="articleBody"]',
                '.article-body'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove unwanted elements
                    for unwanted in content_elem.select('script, style, .ads, .advertisement, .widget, .embed, .related, strong.read'):
                        unwanted.decompose()
                    
                    # Extract text from paragraphs
                    paragraphs = content_elem.find_all(['p', 'div.text'])
                    content_parts = []
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if text and len(text) > 20:  # Filter short text
                            # Skip unwanted patterns
                            skip_patterns = [
                                'Baca Juga:', 
                                'BACA JUGA:', 
                                'Berita Lainnya:',
                                'Halaman Selanjutnya',
                                'BOLASPORT.COM',
                                'Dapatkan Berita',
                                'Download aplikasi'
                            ]
                            if any(skip in text for skip in skip_patterns):
                                continue
                            content_parts.append(text)
                    
                    content = '\n\n'.join(content_parts)
                    break
            
            # Extract author
            author = ""
            author_selectors = [
                '.author-name',
                '.text-author a',
                'span[itemprop="name"]',
                '.article-author',
                '.writer-name'
            ]
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem:
                    author = author_elem.get_text(strip=True)
                    # Clean author name
                    author = re.sub(r'^(Penulis|Editor|Reporter)\s*:\s*', '', author, flags=re.IGNORECASE)
                    break
            
            # Extract date
            date = ""
            date_selectors = [
                '.text-date',
                'time[datetime]',
                '.article-date',
                'span[itemprop="datePublished"]',
                '.publish-date'
            ]
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    # Try to get datetime attribute first
                    date = date_elem.get('datetime', '') or date_elem.get_text(strip=True)
                    break
            
            # Extract category/tags
            tags = []
            # BolaSport biasanya punya kategori di breadcrumb atau tag section
            tag_selectors = [
                '.breadcrumb a',
                '.article-tags a',
                '.tag-item a',
                'meta[property="article:tag"]'
            ]
            for selector in tag_selectors:
                if selector.startswith('meta'):
                    tag_elements = soup.select(selector)
                    for tag_elem in tag_elements:
                        tag_text = tag_elem.get('content', '')
                        if tag_text:
                            tags.append(tag_text)
                else:
                    tag_elements = soup.select(selector)
                    for tag_elem in tag_elements:
                        tag_text = tag_elem.get_text(strip=True)
                        if tag_text and tag_text not in ['Home', 'Beranda', 'BolaSport']:
                            tags.append(tag_text)
            
            # Remove duplicate tags
            tags = list(dict.fromkeys(tags))
            
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
                'source': 'BolaSport',
                'word_count': len(content.split()) if content else 0,
                'char_count': len(content) if content else 0
            }
            
            print(f"✅ Successfully scraped: {title[:50]}...")
            return result
            
        except Exception as e:
            print(f"❌ Error scraping BolaSport: {e}")
            raise Exception(f"Gagal mengambil konten dari BolaSport: {str(e)}")

    def clean_text(self, text):
        text = re.sub(r'ADVERTISEMENT', '', text, flags=re.IGNORECASE)
        text = re.sub(r'SCROLL TO CONTINUE WITH CONTENT', '', text, flags=re.IGNORECASE)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\/]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def validate_url_category(self, url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path.lower()
        
        # Check DetikSport
        if 'detik.com' in domain:
            allowed_paths = self.supported_categories['detik.com']['allowed_paths']
            
            # Check if URL contains any of the allowed paths
            is_valid = any(allowed_path in path for allowed_path in allowed_paths)
            
            if not is_valid:
                sports_list = ', '.join(self.supported_categories['detik.com']['sports'])
                return False, (
                    f"DetikSport, karena aplikasi ini tidak mendukung kategori tersebut. \n\n"
                    f"Hanya artikel dari kategori berikut yang dapat diproses:\n"
                    f"• Sepakbola, Basket, Badminton\n\n"
                    f"Contoh URL yang valid:\n"
                    f"• https://sport.detik.com/sepakbola/...\n"
                    f"• https://sport.detik.com/basket/...\n"
                    f"• https://sport.detik.com/raket/..."
                )
            return True, None
        
        # Check Kompas
        elif 'kompas.com' in domain:
            # bola.kompas.com is always valid (sepakbola)
            if 'bola.kompas.com' in domain:
                return True, None
            
            # For www.kompas.com, check specific paths
            allowed_paths = self.supported_categories['kompas.com']['allowed_paths']
            is_valid = any(allowed_path in path for allowed_path in allowed_paths)
            
            if not is_valid:
                sports_list = '\n• '.join(self.supported_categories['kompas.com']['sports'])
                return False, (
                    f"Kompas, karena aplikasi ini tidak mendukung kategori tersebut.\n\n"
                    f"Hanya artikel dari kategori berikut yang dapat diproses:\n"
                    f"• Sepakbola, Basket, Badminton\n\n"
                    f"Contoh URL yang valid:\n"
                    f"• https://bola.kompas.com/...\n"
                    f"• https://www.kompas.com/sports/...\n"
                    f"• https://www.kompas.com/badminton/..."
                )
            return True, None
        
        # Check BolaSport - accepts all sports
        elif 'bolasport.com' in domain:
            return True, None
        
        # Unsupported domain
        else:
            return False, (
                f"❌ Website tidak didukung!\n\n"
                f"Saat ini hanya mendukung:\n"
                f"• DetikSport (sport.detik.com)\n"
                f"• Kompas Sport (bola.kompas.com, kompas.com)\n"
                f"• BolaSport (bolasport.com)"
            )
    
    def scrape_url(self, url):
        # Validate URL category first
        is_valid, error_message = self.validate_url_category(url)
        
        if not is_valid:
            raise ValueError(error_message)
        
        # Proceed with scraping based on domain
        parsed_url = urlparse(url)
        
        if 'detik.com' in parsed_url.netloc:
            return self.scrape_detiksport(url)
        elif 'kompas.com' in parsed_url.netloc:
            return self.scrape_kompas(url)
        elif 'bolasport.com' in parsed_url.netloc: 
            return self.scrape_bolasport(url)
        else:
            raise ValueError("Website tidak didukung.")
    
# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model instance
ner_model = None
web_scraper = None

def initialize_model():
    """Initialize model dari Hugging Face Hub"""
    global ner_model, web_scraper
    try:
        print("🔄 Downloading model from Hugging Face Hub...")
        
        # Download model dari Hugging Face (PERUBAHAN UTAMA)
        model_path = hf_hub_download(
            repo_id="george121212afasf/model",  # Ganti dengan repo ID Anda
            filename="best_model.pt"
        )
        
        print(f"✅ Model downloaded to: {model_path}")
        print(f"🔄 Loading model into memory...")
        
        ner_model = NERInference(model_path, device='cpu')
        
        # Initialize web scraper
        web_scraper = WebScraper()
        
        print("✅ Model and web scraper initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        import traceback
        traceback.print_exc()  # Print full error untuk debugging
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
                return jsonify({'error': f'Maaf, Gagal mengambil konten dari URL: {str(scrape_error)}'}), 400
        
        print(f"\n🔍 Processing text ({len(text)} characters, {len(text.split())} words)")
        
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
        
        print(f"✅ API response prepared successfully!\n")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error in extract_entities: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': ner_model is not None,
        'scraper_loaded': web_scraper is not None,
        'supported_sources': ['DetikSport', 'Kompas Sport', 'BolaSport'],
        'available_labels': list(ner_model.id2label.values()) if ner_model else []
    })

if __name__ == '__main__':
    print("🚀 Starting SportExtract Flask Server...")
    print("📰 Supported Sources: DetikSport, Kompas Sport, BolaSport")
    
    # Initialize model
    if not initialize_model():
        print("❌ Failed to initialize model. Exiting...")
        sys.exit(1)
    
    # Create templates directory if not exists
    os.makedirs('templates', exist_ok=True)
    
    # Copy website.html to templates (jika ada di root)
    import shutil
    if os.path.exists('website.html'):
        try:
            shutil.copy('website.html', 'templates/website.html')
            print("✅ Website template copied to templates/")
        except Exception as e:
            print(f"⚠️ Could not copy website.html: {e}")
    
    # Hugging Face Spaces menggunakan port 7860
    port = int(os.environ.get("PORT", 7860))
    print(f"🌐 Server starting on port {port}")
    
    # Debug mode OFF untuk production
    app.run(debug=False, host='0.0.0.0', port=port)