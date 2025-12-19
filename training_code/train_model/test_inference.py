#!/usr/bin/env python3
"""
Script untuk testing inference model IndoBERT NER pada teks berita
Interactive testing dan demo model
"""

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from typing import List, Dict

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
        """Prediksi entities dari teks dengan support untuk teks panjang"""
        # Check text length first
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        print(f"📊 Text analysis: {len(tokens)} tokens, {len(text)} characters, {len(text.split())} words")
        
        if len(tokens) <= 510:  # 512 - 2 special tokens
            # Text fits in single window
            return self._predict_single_window(text, threshold)
        else:
            # Text too long, use sliding window approach
            print(f"⚠️  Text too long ({len(tokens)} tokens), using sliding window approach...")
            return self._predict_sliding_window(text, threshold)
    
    def _predict_single_window(self, text, threshold=0.5):
        """Prediksi entities untuk teks yang muat dalam satu window"""
        # Tokenize with proper handling
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,  # Enable truncation for safety
            max_length=512,   # Set max length
            padding='max_length'  # Add padding for consistency
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
    
    def _predict_sliding_window(self, text, threshold=0.5, window_size=400, overlap=50):
        """Prediksi entities untuk teks panjang dengan sliding window"""
        words = text.split()
        all_entities = []
        
        print(f"🔄 Processing text with sliding window (window_size={window_size}, overlap={overlap})")
        
        for i in range(0, len(words), window_size - overlap):
            # Create window
            window_words = words[i:i + window_size]
            window_text = " ".join(window_words)
            
            # Calculate actual position in original text
            start_pos = len(" ".join(words[:i]))
            if i > 0:
                start_pos += 1  # Account for space
            
            print(f"   Processing window {i//(window_size-overlap)+1}: words {i}-{min(i+window_size, len(words))}")
            
            # Predict on this window
            window_entities = self._predict_single_window(window_text, threshold)
            
            # Adjust entity positions to match original text
            for entity in window_entities:
                entity['start'] += start_pos
                entity['end'] += start_pos
                # Re-extract text from original position
                entity['text'] = text[entity['start']:entity['end']]
            
            all_entities.extend(window_entities)
        
        # Remove duplicate entities (same text and position)
        unique_entities = self._remove_duplicate_entities(all_entities)
        
        print(f"✅ Found {len(unique_entities)} unique entities across all windows")
        return unique_entities
    
    def _remove_duplicate_entities(self, entities):
        """Remove duplicate entities based on text and position"""
        unique_entities = []
        seen = set()
        
        for entity in entities:
            # Create a key based on text, start, end, and labels
            key = (entity['text'].strip().lower(), entity['start'], entity['end'], tuple(sorted(entity['labels'])))
            
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def predict_entities_unlimited(self, text, threshold=0.5):
        """Prediksi entities untuk teks tanpa batasan panjang dengan smart chunking"""
        print(f"🚀 Processing unlimited length text...")
        print(f"📊 Text stats: {len(text)} characters, {len(text.split())} words")
        
        # Check if text is actually long
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= 510:
            print(f"📝 Text fits in single window, using standard processing...")
            return self._predict_single_window(text, threshold)
        
        # Split text into sentences for better chunking
        sentences = self._split_into_sentences(text)
        print(f"📝 Split into {len(sentences)} sentences")
        
        all_entities = []
        current_chunk = ""
        chunk_count = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed token limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            tokens = self.tokenizer.encode(test_chunk, add_special_tokens=False)
            
            if len(tokens) <= 510:  # Can fit in current chunk
                current_chunk = test_chunk
            else:
                # Process current chunk if it has content
                if current_chunk.strip():
                    chunk_count += 1
                    print(f"🔄 Processing chunk {chunk_count}: {len(current_chunk)} chars, {len(current_chunk.split())} words")
                    chunk_entities = self._predict_single_window(current_chunk, threshold)
                    all_entities.extend(chunk_entities)
                    print(f"   Found {len(chunk_entities)} entities in this chunk")
                
                # Start new chunk with current sentence
                current_chunk = sentence
        
        # Process the last chunk
        if current_chunk.strip():
            chunk_count += 1
            print(f"🔄 Processing final chunk {chunk_count}: {len(current_chunk)} chars, {len(current_chunk.split())} words")
            chunk_entities = self._predict_single_window(current_chunk, threshold)
            all_entities.extend(chunk_entities)
            print(f"   Found {len(chunk_entities)} entities in final chunk")
        
        # Remove duplicates and merge overlapping entities
        print(f"🔄 Removing duplicates from {len(all_entities)} total entities...")
        unique_entities = self._remove_duplicate_entities(all_entities)
        print(f"🔄 Merging overlapping entities...")
        merged_entities = self._merge_overlapping_entities(unique_entities, text)
        
        print(f"✅ Final result: {len(merged_entities)} unique entities from {chunk_count} chunks")
        return merged_entities
    
    def _split_into_sentences(self, text):
        """Split text into sentences for better chunking"""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _merge_overlapping_entities(self, entities, text):
        """Merge entities that overlap or are adjacent"""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x['start'])
        merged = []
        
        for entity in entities:
            if not merged:
                merged.append(entity)
            else:
                last_entity = merged[-1]
                
                # Check if entities overlap or are very close
                if (entity['start'] <= last_entity['end'] + 2 and 
                    set(entity['labels']) == set(last_entity['labels'])):
                    # Merge entities
                    last_entity['end'] = max(last_entity['end'], entity['end'])
                    last_entity['text'] = text[last_entity['start']:last_entity['end']]
                else:
                    merged.append(entity)
        
        return merged
    
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
    
    def format_output(self, text, entities):
        """Format output dengan grouping per label"""
        if not entities:
            return {}, "No entities detected"
        
        # Group entities by label
        label_groups = {}
        
        for entity in entities:
            for label in entity['labels']:
                if label not in label_groups:
                    label_groups[label] = []
                
                # Add unique text to avoid duplicates
                entity_text = entity['text'].strip()
                if entity_text and entity_text not in label_groups[label]:
                    label_groups[label].append(entity_text)
        
        return label_groups, "Entities grouped by label"
    
    def print_formatted_output(self, label_groups):
        """Print output dalam format yang diminta"""
        if not label_groups:
            print("No entities detected")
            return
        
        # Sort labels for consistent output
        for label in sorted(label_groups.keys()):
            entities = label_groups[label]
            # Filter out very short words (less than 2 characters)
            filtered_entities = [e for e in entities if len(e) > 2]
            
            if filtered_entities:
                # Join multiple entities with comma
                entities_str = ", ".join(filtered_entities)
                print(f"{label}: {entities_str}")
    
    def test_sample_texts(self):
        """Test dengan sample teks berita"""
        sample_texts = [
            "Lionel Messi mencetak gol untuk Argentina dalam pertandingan melawan Brasil di Stadion Maracana pada tanggal 15 Juli 2023.",
            "Cristiano Ronaldo yang berusia 38 tahun memenangkan Ballon d'Or untuk kelima kalinya dalam kariernya.",
            "Manchester United mengalahkan Liverpool dengan skor 2-1 dalam derby Inggris di Old Trafford.",
            "Kylian Mbappe dari Prancis mencetak hat-trick dalam pertandingan melawan Jerman di Piala Dunia 2022.",
            "Real Madrid mengangkat trofi Liga Champions setelah mengalahkan Bayern Munich di final.",
            # Test dengan teks panjang
            """Jakarta - Juventus dan AC Milan tuntas tanpa pemenang dalam lanjutan Liga Italia. Adrien Rabiot marah karena Rossoneri tak mampu menumbangkan Bianconeri. Juventus vs Milan berlangsung di Allianz Stadium, Senin (6/10/2025) dini hari WIB. Pertandingan ini berakhir dengan skor 0-0. Milan sebetulnya mampu melepaskan 13 percobaan dengan empat yang mengarah ke gawang. Juventus ada tiga percobaan ke gawang dari 12 upaya.Pasukan Massimiliano Allegri sempat mendapatkan hadiah penalti di babak kedua. Christian Pulisic, yang menjadi eksekutor, gagal memasukkan bola ke gawang. Rabiot tak senang dengan hasil imbang ini. Mantan gelandang Juventus itu merasa timnya harusnya bisa menang. "Malam ini sangat emosional. Saya kenal banyak pemain di Juventus, stadion, staf, bahkan Tudor Saya sangat senang bisa kembali ke sini. Saya ingin menang, tapi begitulah sepakbola - kami akan terus melanjutkan perjalanan kami," kata Rabiot kepada DAZN. "Saya marah karena seharusnya kami menang. Kami punya peluang, tapi ada yang kurang. Kami harus berbuat lebih banyak dan meningkatkan diri sebagai tim. Ini baru awal musim, tapi menang di sini akan sangat penting," tegasnya. Juventus duduk di posisi kelima dengan 12 poin. Milan di urutan ketiga dengan 13 poin."""
        ]
        
        print(f"\n🧪 Testing dengan sample teks berita:")
        print("="*60)
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n📰 Sample {i}:")
            print(f"Text: {text}")
            
            entities = self.predict_entities(text)
            label_groups, _ = self.format_output(text, entities)
            
            print(f"Entities detected: {len(entities)}")
            print("\n🎯 Detected Entities:")
            self.print_formatted_output(label_groups)
            
            print("-" * 60)
    
    def test_long_text(self):
        """Test khusus untuk teks panjang"""
        long_text = """Jakarta - Juventus dan AC Milan tuntas tanpa pemenang dalam lanjutan Liga Italia. Adrien Rabiot marah karena Rossoneri tak mampu menumbangkan Bianconeri. Juventus vs Milan berlangsung di Allianz Stadium, Senin (6/10/2025) dini hari WIB. Pertandingan ini berakhir dengan skor 0-0. Milan sebetulnya mampu melepaskan 13 percobaan dengan empat yang mengarah ke gawang. Juventus ada tiga percobaan ke gawang dari 12 upaya.Pasukan Massimiliano Allegri sempat mendapatkan hadiah penalti di babak kedua. Christian Pulisic, yang menjadi eksekutor, gagal memasukkan bola ke gawang. Rabiot tak senang dengan hasil imbang ini. Mantan gelandang Juventus itu merasa timnya harusnya bisa menang. "Malam ini sangat emosional. Saya kenal banyak pemain di Juventus, stadion, staf, bahkan Tudor Saya sangat senang bisa kembali ke sini. Saya ingin menang, tapi begitulah sepakbola - kami akan terus melanjutkan perjalanan kami," kata Rabiot kepada DAZN. "Saya marah karena seharusnya kami menang. Kami punya peluang, tapi ada yang kurang. Kami harus berbuat lebih banyak dan meningkatkan diri sebagai tim. Ini baru awal musim, tapi menang di sini akan sangat penting," tegasnya. Juventus duduk di posisi kelima dengan 12 poin. Milan di urutan ketiga dengan 13 poin."""
        
        print(f"\n🧪 Testing LONG TEXT:")
        print("="*80)
        print(f"Text length: {len(long_text)} characters, {len(long_text.split())} words")
        print(f"Text preview: {long_text[:200]}...")
        print("="*80)
        
        # Test dengan standard method
        print(f"\n📊 Standard Method:")
        entities_standard = self.predict_entities(long_text)
        label_groups_standard, _ = self.format_output(long_text, entities_standard)
        print(f"✅ Entities detected: {len(entities_standard)}")
        print("\n🎯 Detected Entities:")
        self.print_formatted_output(label_groups_standard)
        
        print("\n" + "="*80)
        
        # Test dengan unlimited method
        print(f"\n📊 Unlimited Method:")
        entities_unlimited = self.predict_entities_unlimited(long_text)
        label_groups_unlimited, _ = self.format_output(long_text, entities_unlimited)
        print(f"✅ Entities detected: {len(entities_unlimited)}")
        print("\n🎯 Detected Entities:")
        self.print_formatted_output(label_groups_unlimited)
        
        print("="*80)

def main():
    """Main function"""
    print("="*80)
    print("TESTING INFERENCE MODEL INDOBERT NER")
    print("="*80)
    
    # Configuration
    model_path = './outputs_indobert_multilabel/best_model.pt'
    device = 'cpu'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first or check the path.")
        return
    
    # Create inference object
    try:
        ner_inference = NERInference(model_path, device)
    except Exception as e:
        print(f"❌ Error creating inference object: {e}")
        return
    
    # Test with sample texts
    ner_inference.test_sample_texts()
    
    # Test with long text
    ner_inference.test_long_text()
    
    # Interactive testing
    print(f"\n🎯 Interactive Testing:")
    print("Commands:")
    print("  'quit' or 'exit' - Exit program")
    print("  'unlimited' - Toggle unlimited text processing")
    print("  'file' - Load text from file")
    print("  'multiline' - Enter multiline text (type 'END' on new line to finish)")
    print("-" * 60)
    
    unlimited_mode = False
    
    while True:
        try:
            print(f"\n📝 Mode: {'UNLIMITED' if unlimited_mode else 'STANDARD'}")
            user_input = input("Enter text or command: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'unlimited':
                unlimited_mode = not unlimited_mode
                print(f"🔄 Switched to {'UNLIMITED' if unlimited_mode else 'STANDARD'} mode")
                continue
            
            if user_input.lower() == 'file':
                file_path = input("Enter file path: ").strip()
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        user_input = f.read()
                    print(f"✅ Loaded text from file: {len(user_input)} characters")
                except Exception as e:
                    print(f"❌ Error loading file: {e}")
                    continue
            
            elif user_input.lower() == 'multiline':
                print("Enter your text (type 'END' on a new line to finish):")
                lines = []
                while True:
                    line = input()
                    if line.strip().upper() == 'END':
                        break
                    lines.append(line)
                user_input = '\n'.join(lines)
                print(f"✅ Multiline text entered: {len(user_input)} characters")
            
            if not user_input:
                continue
            
            print(f"\n🔍 Analyzing text...")
            print(f"📊 Text length: {len(user_input)} characters, {len(user_input.split())} words")
            
            # Show text preview (first 200 chars)
            if len(user_input) > 200:
                print(f"📄 Text preview: {user_input[:200]}...")
            else:
                print(f"📄 Full text: {user_input}")
            
            # Choose prediction method based on mode
            if unlimited_mode:
                entities = ner_inference.predict_entities_unlimited(user_input)
            else:
                entities = ner_inference.predict_entities(user_input)
            
            label_groups, _ = ner_inference.format_output(user_input, entities)
            
            print(f"✅ Entities detected: {len(entities)}")
            print("\n🎯 Detected Entities:")
            ner_inference.print_formatted_output(label_groups)
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()