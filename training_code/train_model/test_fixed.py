import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_fscore_support
from typing import List, Dict, Tuple
import re

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

class FixedNERValidator:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.label2id = None
        
        # Load model dan tokenizer
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load model dari checkpoint"""
        print(f"🔄 Loading model from: {model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model config
            config = checkpoint.get('config', {})
            model_name = config.get('model_name', 'indobenchmark/indobert-base-p1')
            self.id2label = checkpoint.get('id2label', {})
            self.label2id = {v: k for k, v in self.id2label.items()}
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create model
            self.model = IndoBERTNERModelMultiLabel(model_name, len(self.id2label))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully!")
            print(f"   Model type: {checkpoint.get('model_type', 'Unknown')}")
            print(f"   F1 Score: {checkpoint.get('f1_score', 'Unknown'):.4f}")
            print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"   Labels: {list(self.id2label.values())}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_with_smart_grouping(self, text, threshold=0.5):
        """Prediksi dengan smart grouping untuk ALASAN_PERISTIWA"""
        # Tokenize dengan alignment yang lebih baik
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            padding='max_length',
            add_special_tokens=True
        )
        
        input_ids = torch.tensor([encoding['input_ids']], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([encoding['attention_mask']], dtype=torch.long).to(self.device)
        offset_mapping = encoding['offset_mapping']
        
        # Predict dengan threshold yang optimal
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            predictions = torch.sigmoid(logits) > threshold
        
        # Extract entities dengan smart grouping
        entities = self.extract_entities_with_smart_grouping(
            text, predictions[0], offset_mapping, attention_mask[0], threshold
        )
        
        return entities
    
    def extract_entities_with_smart_grouping(self, text, predictions, offset_mapping, attention_mask, threshold):
        """Extract entities dengan smart grouping khusus untuk ALASAN_PERISTIWA"""
        entities = []
        
        # Step 1: Get all token predictions
        token_predictions = []
        for token_idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == token_end or attention_mask[token_idx] == 0:
                continue
            
            # Get active labels for this token
            active_labels = set()
            for label_id, is_active in enumerate(predictions[token_idx]):
                if is_active:
                    label_name = self.id2label.get(label_id, f"LABEL_{label_id}")
                    active_labels.add(label_name)
            
            token_predictions.append({
                'start': token_start,
                'end': token_end,
                'labels': active_labels,
                'text': text[token_start:token_end]
            })
        
        # Step 2: Group consecutive tokens with same labels
        current_entity = None
        current_labels = set()
        
        for token_pred in token_predictions:
            active_labels = token_pred['labels']
            
            # Check if this token continues the current entity
            if current_entity is not None and active_labels == current_labels:
                # Continue current entity
                current_entity['end'] = token_pred['end']
                current_entity['text'] = text[current_entity['start']:current_entity['end']]
            else:
                # Save previous entity if exists
                if current_entity is not None:
                    entities.append(current_entity)
                
                # Start new entity if has labels
                if active_labels:
                    current_entity = {
                        'text': token_pred['text'],
                        'start': token_pred['start'],
                        'end': token_pred['end'],
                        'labels': list(active_labels)
                    }
                    current_labels = active_labels
                else:
                    current_entity = None
                    current_labels = set()
        
        # Add last entity
        if current_entity is not None:
            entities.append(current_entity)
        
        # Step 3: Smart merging untuk ALASAN_PERISTIWA
        entities = self.smart_merge_alasan_peristiwa(entities, text)
        
        return entities
    
    def smart_merge_alasan_peristiwa(self, entities, text):
        """Smart merging untuk ALASAN_PERISTIWA yang berdekatan"""
        if not entities:
            return entities
        
        merged_entities = []
        i = 0
        
        while i < len(entities):
            current_entity = entities[i]
            
            # Check if current entity has ALASAN_PERISTIWA
            if 'ALASAN_PERISTIWA' in current_entity['labels']:
                # Look ahead for nearby ALASAN_PERISTIWA entities
                j = i + 1
                while j < len(entities):
                    next_entity = entities[j]
                    
                    # Check if next entity has ALASAN_PERISTIWA and is close
                    if ('ALASAN_PERISTIWA' in next_entity['labels'] and 
                        next_entity['start'] - current_entity['end'] <= 5):  # Max 5 chars gap
                        
                        # Merge entities
                        current_entity['end'] = next_entity['end']
                        current_entity['text'] = text[current_entity['start']:current_entity['end']]
                        
                        # Merge labels (keep unique labels)
                        all_labels = set(current_entity['labels'] + next_entity['labels'])
                        current_entity['labels'] = list(all_labels)
                        
                        j += 1
                    else:
                        break
                
                merged_entities.append(current_entity)
                i = j
            else:
                merged_entities.append(current_entity)
                i += 1
        
        return merged_entities
    
    def normalize_entities(self, entities, text):
        """Normalize entities untuk perbandingan yang adil"""
        normalized = []
        
        for entity in entities:
            if isinstance(entity, dict):
                # Extract entity information
                entity_text = entity.get('text', '').strip()
                start = entity.get('start', 0)
                end = entity.get('end', start + len(entity_text))
                labels = entity.get('labels', [])
                
                if entity_text and labels:
                    # Clean entity text
                    entity_text = entity_text.strip()
                    
                    # Create normalized entity
                    normalized_entity = {
                        'text': entity_text.lower(),
                        'start': start,
                        'end': end,
                        'labels': sorted(labels) if isinstance(labels, list) else [labels]
                    }
                    normalized.append(normalized_entity)
        
        return normalized
    
    def calculate_entity_metrics(self, gt_entities, pred_entities):
        """Calculate metrics berbasis entitas (bukan subword)"""
        print(f"   Calculating entity-based metrics...")
        print(f"   Ground truth entities: {len(gt_entities)}")
        print(f"   Predicted entities: {len(pred_entities)}")
        
        # Calculate per-label metrics
        label_metrics = {}
        
        for label_name in self.id2label.values():
            # Filter entities by label
            gt_label_entities = [e for e in gt_entities if label_name in e['labels']]
            pred_label_entities = [e for e in pred_entities if label_name in e['labels']]
            
            # Calculate precision, recall, F1 for this label
            precision, recall, f1 = self.calculate_entity_precision_recall_f1(
                gt_label_entities, pred_label_entities
            )
            
            label_metrics[label_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': len(gt_label_entities),
                'predicted_count': len(pred_label_entities)
            }
        
        # Calculate overall metrics
        total_gt = len(gt_entities)
        total_pred = len(pred_entities)
        
        # Overall precision, recall, F1
        overall_precision, overall_recall, overall_f1 = self.calculate_entity_precision_recall_f1(
            gt_entities, pred_entities
        )
        
        return {
            'micro_f1': overall_f1,
            'macro_f1': np.mean([m['f1'] for m in label_metrics.values()]),
            'weighted_f1': np.average([m['f1'] for m in label_metrics.values()], 
                                     weights=[m['support'] for m in label_metrics.values()]),
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'total_gt_entities': total_gt,
            'total_pred_entities': total_pred,
            'label_metrics': label_metrics
        }
    
    def calculate_entity_precision_recall_f1(self, gt_entities, pred_entities):
        """Calculate precision, recall, and F1 for entities"""
        if not pred_entities:
            return 0.0, 0.0, 0.0
        
        # Find matches between ground truth and predictions
        matches = 0
        used_pred_indices = set()
        
        for gt_entity in gt_entities:
            best_match = None
            best_score = 0
            best_pred_idx = -1
            
            for pred_idx, pred_entity in enumerate(pred_entities):
                if pred_idx in used_pred_indices:
                    continue
                
                # Calculate match score
                score = self.calculate_entity_match_score(gt_entity, pred_entity)
                
                if score > best_score:
                    best_score = score
                    best_match = pred_entity
                    best_pred_idx = pred_idx
            
            # If we found a good match (score > 0.5), count it
            if best_score > 0.5:
                matches += 1
                used_pred_indices.add(best_pred_idx)
        
        # Calculate metrics
        precision = matches / len(pred_entities) if pred_entities else 0.0
        recall = matches / len(gt_entities) if gt_entities else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def calculate_entity_match_score(self, gt_entity, pred_entity):
        """Calculate match score between two entities"""
        # Text similarity
        gt_text = gt_entity['text']
        pred_text = pred_entity['text']
        
        # Exact text match
        if gt_text == pred_text:
            return 1.0
        
        # Label overlap
        gt_labels = set(gt_entity['labels'])
        pred_labels = set(pred_entity['labels'])
        
        if not gt_labels.intersection(pred_labels):
            return 0.0
        
        # Position overlap
        gt_start, gt_end = gt_entity['start'], gt_entity['end']
        pred_start, pred_end = pred_entity['start'], pred_entity['end']
        
        # Calculate overlap
        overlap_start = max(gt_start, pred_start)
        overlap_end = min(gt_end, pred_end)
        overlap_length = max(0, overlap_end - overlap_start)
        
        gt_length = gt_end - gt_start
        pred_length = pred_end - pred_start
        
        # Jaccard similarity for positions
        union_length = gt_length + pred_length - overlap_length
        position_score = overlap_length / union_length if union_length > 0 else 0.0
        
        # Text similarity (simple)
        text_score = 0.0
        if gt_text in pred_text or pred_text in gt_text:
            text_score = 0.5
        elif len(set(gt_text.split()) & set(pred_text.split())) > 0:
            text_score = 0.3
        
        # Combined score
        total_score = (position_score * 0.6 + text_score * 0.4)
        
        return min(total_score, 1.0)
    
    def convert_annotations_to_char_labels(self, annotations, text):
        """Convert annotations to character-level labels"""
        char_labels = []
        
        # Initialize all characters as 'O' (no entity)
        for _ in range(len(text)):
            char_labels.append('O')
        
        # Fill in entity labels
        for annotation in annotations:
            if 'text' in annotation and 'labels' in annotation:
                start = annotation.get('start', 0)
                end = annotation.get('end', start + len(annotation['text']))
                entity_labels = annotation['labels']
                
                # Ensure bounds are valid
                start = max(0, min(start, len(text)))
                end = max(start, min(end, len(text)))
                
                # Assign labels to characters
                for i in range(start, end):
                    if i < len(char_labels):
                        # Use the first label if multiple labels
                        label = entity_labels[0] if isinstance(entity_labels, list) else entity_labels
                        char_labels[i] = label
        
        return char_labels
    
    def convert_entities_to_char_labels(self, entities, text):
        """Convert predicted entities to character-level labels"""
        char_labels = []
        
        # Initialize all characters as 'O' (no entity)
        for _ in range(len(text)):
            char_labels.append('O')
        
        # Fill in entity labels
        for entity in entities:
            start = entity.get('start', 0)
            end = entity.get('end', start + len(entity.get('text', '')))
            entity_labels = entity.get('labels', [])
            
            # Ensure bounds are valid
            start = max(0, min(start, len(text)))
            end = max(start, min(end, len(text)))
            
            # Assign labels to characters
            for i in range(start, end):
                if i < len(char_labels) and entity_labels:
                    # Use the first label if multiple labels
                    label = entity_labels[0] if isinstance(entity_labels, list) else entity_labels
                    char_labels[i] = label
        
        return char_labels
    
    def calculate_char_level_metrics(self, gt_labels, pred_labels):
        """Calculate metrics berbasis karakter (character-level)"""
        print(f"   Calculating character-level metrics...")
        print(f"   Ground truth characters: {len(gt_labels)}")
        print(f"   Predicted characters: {len(pred_labels)}")
        
        # Ensure same length
        min_length = min(len(gt_labels), len(pred_labels))
        gt_labels = gt_labels[:min_length]
        pred_labels = pred_labels[:min_length]
        
        # Convert to numpy arrays
        gt_labels = np.array(gt_labels)
        pred_labels = np.array(pred_labels)
        
        # Calculate overall metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(gt_labels, pred_labels)
        
        # Calculate per-label metrics
        unique_labels = list(self.id2label.values()) + ['O']
        precision, recall, f1, support = precision_recall_fscore_support(
            gt_labels, pred_labels, labels=unique_labels, average=None, zero_division=0
        )
        
        # Overall F1 scores
        micro_f1 = f1_score(gt_labels, pred_labels, average='micro', zero_division=0)
        macro_f1 = f1_score(gt_labels, pred_labels, average='macro', zero_division=0)
        weighted_f1 = f1_score(gt_labels, pred_labels, average='weighted', zero_division=0)
        
        # Per-label metrics
        label_metrics = {}
        for i, label in enumerate(unique_labels):
            if i < len(precision):
                label_metrics[label] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
        
        return {
            'accuracy': float(accuracy),
            'micro_f1': float(micro_f1),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'total_characters': min_length,
            'label_metrics': label_metrics
        }
    
    def evaluate_on_dataset_fixed(self, data_path, threshold=0.5, sample_size=None):
        """Evaluasi model dengan perbaikan character-level evaluation"""
        print(f"\n📊 Evaluating model on dataset: {data_path}")
        print(f"   Using threshold: {threshold}")
        print(f"   Evaluation method: Character-level (not subword-based)")
        
        # Load data
        df = pd.read_csv(data_path)
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        print(f"   Dataset size: {len(df)} samples")
        
        # Character-level evaluation
        all_gt_labels = []
        all_pred_labels = []
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            if pd.isna(row['content']) or pd.isna(row['label']):
                continue
            
            text = str(row['content'])
            if len(text) < 10:
                continue
            
            # Parse ground truth
            try:
                gt_annotations = json.loads(str(row['label']))
            except:
                continue
            
            # Predict with smart grouping
            predicted_entities = self.predict_with_smart_grouping(text, threshold)
            
            # Convert to character-level labels
            gt_char_labels = self.convert_annotations_to_char_labels(gt_annotations, text)
            pred_char_labels = self.convert_entities_to_char_labels(predicted_entities, text)
            
            # Ensure same length
            min_length = min(len(gt_char_labels), len(pred_char_labels), len(text))
            gt_char_labels = gt_char_labels[:min_length]
            pred_char_labels = pred_char_labels[:min_length]
            
            all_gt_labels.extend(gt_char_labels)
            all_pred_labels.extend(pred_char_labels)
            
            # Store results
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'ground_truth': gt_annotations,
                'predictions': predicted_entities,
                'gt_char_labels': gt_char_labels,
                'pred_char_labels': pred_char_labels
            })
        
        # Calculate character-level metrics
        metrics = self.calculate_char_level_metrics(all_gt_labels, all_pred_labels)
        
        return results, metrics
    
    def save_results_fixed(self, results, metrics, threshold, output_dir='./validation_results_fixed'):
        """Save hasil evaluasi dengan perbaikan"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics summary
        with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save sample predictions
        sample_results = results[:10]
        with open(os.path.join(output_dir, 'sample_predictions.json'), 'w') as f:
            json.dump(sample_results, f, indent=2, ensure_ascii=False)
        
        # Save detailed report
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("INDOBERT NER MODEL EVALUATION REPORT (FIXED)\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Threshold: {threshold}\n\n")
            
            f.write(f"Overall Metrics (Character-level):\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Micro F1: {metrics['micro_f1']:.4f}\n")
            f.write(f"  Macro F1: {metrics['macro_f1']:.4f}\n")
            f.write(f"  Weighted F1: {metrics['weighted_f1']:.4f}\n")
            f.write(f"  Total Characters: {metrics['total_characters']}\n\n")
            
            f.write("Per-Label Metrics:\n")
            for label, label_metrics in metrics['label_metrics'].items():
                f.write(f"  {label}:\n")
                f.write(f"    Precision: {label_metrics['precision']:.4f}\n")
                f.write(f"    Recall: {label_metrics['recall']:.4f}\n")
                f.write(f"    F1: {label_metrics['f1']:.4f}\n")
                f.write(f"    Support: {label_metrics['support']}\n\n")
            
            f.write("Sample Predictions:\n")
            for i, result in enumerate(sample_results[:5]):
                f.write(f"\nSample {i+1}:\n")
                f.write(f"Text: {result['text']}\n")
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Predictions: {result['predictions']}\n")
        
        print(f"📊 Results saved to: {output_dir}")

def main():
    """Main function"""
    print("="*80)
    print("VALIDASI MODEL INDOBERT NER - FIXED VERSION")
    print("Perbaikan: Smart grouping untuk ALASAN_PERISTIWA")
    print("="*80)
    
    # Configuration
    model_path = './outputs_indobert_multilabel/best_model.pt'
    data_path = '../test_dataset.csv'
    output_dir = './test_results_fixed'
    device = 'cpu'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first or check the path.")
        return
    
    # Create validator
    try:
        validator = FixedNERValidator(model_path, device)
    except Exception as e:
        print(f"❌ Error creating validator: {e}")
        return
    
    # Test single text inference
    print(f"\n🧪 Testing single text inference...")
    test_text = "Lionel Messi mencetak gol untuk Argentina dalam pertandingan melawan Brasil di Stadion Maracana pada tanggal 15 Juli 2023."
    
    print(f"Text: {test_text}")
    entities = validator.predict_with_smart_grouping(test_text, threshold=0.5)
    print(f"Detected entities: {len(entities)} entities found")
    for i, entity in enumerate(entities):
        print(f"  {i+1}. {entity['text']} -> {entity['labels']}")
    
    # Evaluate with fixed entity extraction
    print(f"\n📊 Evaluating with fixed entity extraction...")
    results, metrics = validator.evaluate_on_dataset_fixed(data_path, threshold=0.5, sample_size=100)
    
    # Print results
    print(f"\n📈 Character-level Evaluation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Micro F1: {metrics['micro_f1']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Total Characters: {metrics['total_characters']}")
    
    print(f"\n📊 Per-Label Results:")
    for label, label_metrics in metrics['label_metrics'].items():
        print(f"  {label}: F1={label_metrics['f1']:.4f}, "
              f"Precision={label_metrics['precision']:.4f}, "
              f"Recall={label_metrics['recall']:.4f}, "
              f"Support={label_metrics['support']}")
    
    # Save results
    validator.save_results_fixed(results, metrics, 0.5, output_dir)
    
    print(f"\n✅ Fixed validation completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
