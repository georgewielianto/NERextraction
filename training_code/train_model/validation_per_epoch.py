#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import re
from typing import List, Dict, Tuple
from datetime import datetime

# Set style untuk matplotlib - DISABLED
# plt.style.use('seaborn-v0_8')
# sns.set_palette("husl")

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

class NERDatasetMultiLabel(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
        return item

def load_and_preprocess_data_multilabel(data_path: str, tokenizer, max_length: int = 512):
    """Load dan preprocess data dengan multi-label encoding"""
    print(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} rows")
    
    texts = []
    annotations_list = []
    
    # Entity type mapping
    entity_types = [
        'TIM', 'ATLET', 'KEJUARAAN', 'STADION', 'TANGGAL', 'SKOR', 
        'AKSI', 'ALASAN_PERISTIWA', 'STATISTIK', 'PENGHARGAAN', 
        'KEWARGANEGARAAN', 'ORGANISASI', 'POSISI', 'UMUR'
    ]
    
    # Create label mapping
    label2id = {}
    for entity_type in entity_types:
        label2id[entity_type] = len(label2id)
    
    id2label = {v: k for k, v in label2id.items()}
    
    print(f"Number of entity types: {len(label2id)}")
    print(f"Label mapping: {label2id}")
    
    print(f"\nParsing annotations from {len(df)} rows...")
    
    for idx, row in df.iterrows():
        if pd.isna(row['content']) or pd.isna(row['label']):
            continue
            
        text = str(row['content'])
        if len(text) < 10:
            continue
            
        # Parse annotations
        try:
            annotations = json.loads(str(row['label']))
        except:
            continue
        
        texts.append(text)
        annotations_list.append(annotations)
        
        if (idx + 1) % 500 == 0:
            print(f"Parsed {idx + 1}/{len(df)} rows")
    
    print(f"\n✅ Parsed {len(texts)} valid texts")
    
    # Encode dengan alignment
    encodings, labels = encode_labels_with_alignment(
        texts, annotations_list, tokenizer, label2id, max_length
    )
    
    return encodings, labels, label2id, id2label

def encode_labels_with_alignment(texts, annotations_list, tokenizer, label2id, max_length):
    """Encode texts dan align labels"""
    num_labels = len(label2id)
    
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    print(f"\nEncoding {len(texts)} samples with alignment...")
    
    for idx, (text, annotations) in enumerate(zip(texts, annotations_list)):
        # Tokenize dengan offset_mapping
        try:
            encoding = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
        except Exception as e:
            print(f"Error tokenizing sample {idx}: {e}")
            continue
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        offset_mapping = encoding['offset_mapping']
        
        # Create multi-hot labels for each character position
        char_labels = np.zeros((len(text), num_labels), dtype=np.float32)
        
        # Mark all overlapping annotations
        for annotation in annotations:
            if 'text' in annotation and 'labels' in annotation:
                entity_labels = annotation['labels']
                start = annotation.get('start', 0)
                end = annotation.get('end', start + len(annotation['text']))
                
                # Validate span boundaries
                if start < 0 or end > len(text) or start >= end:
                    continue
                
                # Mark all entity types for this span
                for entity_type in entity_labels:
                    if entity_type in label2id:
                        label_id = label2id[entity_type]
                        for i in range(start, min(end, len(text))):
                            if i < len(char_labels):
                                char_labels[i][label_id] = 1.0
        
        # Convert character labels to token labels using offset_mapping
        token_labels = []
        
        for token_start, token_end in offset_mapping:
            if token_start == token_end:
                # Special token ([CLS], [SEP], [PAD])
                token_label = np.zeros(num_labels, dtype=np.float32)
            else:
                # Aggregate labels across character range (max pooling)
                token_label = np.zeros(num_labels, dtype=np.float32)
                for i in range(token_start, min(token_end, len(char_labels))):
                    if i < len(char_labels):
                        token_label = np.maximum(token_label, char_labels[i])
            
            token_labels.append(token_label.tolist())
        
        # Simpan encodings dan labels
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(token_labels)
        
        if (idx + 1) % 500 == 0:
            print(f"Encoded {idx + 1}/{len(texts)} samples")
    
    print(f"\n✅ Encoding completed!")
    print(f"   Valid samples: {len(all_input_ids)}")
    
    # Return as dict untuk consistency
    encodings = {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks
    }
    
    return encodings, all_labels

def evaluate_model(model, dataloader, device, id2label):
    """Evaluate model dan return detailed metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            predictions = torch.sigmoid(logits) > 0.5
            
            # Collect predictions and labels (only non-padded tokens)
            for i in range(len(predictions)):
                seq_len = attention_mask[i].sum().item()
                all_predictions.extend(predictions[i][:seq_len].cpu().numpy())
                all_labels.extend(labels[i][:seq_len].cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics per label
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Calculate overall metrics
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='micro', zero_division=0
    )
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    
    # Create detailed results
    results = {
        'per_label': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        },
        'overall': {
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        },
        'label_names': list(id2label.values())
    }
    
    return results

def create_visualizations(results_per_epoch, output_dir):
    """Create visualizations untuk semua metrics - DISABLED due to matplotlib issues"""
    print("⚠️  Visualizations disabled due to matplotlib compatibility issues")
    print("   Only text report will be generated")
    
    # Create simple CSV files for data analysis
    epochs = list(results_per_epoch.keys())
    label_names = results_per_epoch[epochs[0]]['label_names']
    
    # Save F1 scores to CSV
    f1_data = []
    for epoch in epochs:
        f1_scores = results_per_epoch[epoch]['per_label']['f1']
        row = {'Epoch': epoch}
        for i, label in enumerate(label_names):
            row[label] = f1_scores[i]
        f1_data.append(row)
    
    f1_df = pd.DataFrame(f1_data)
    f1_csv_path = os.path.join(output_dir, 'f1_scores_per_epoch.csv')
    f1_df.to_csv(f1_csv_path, index=False)
    print(f"✅ F1 scores CSV saved: {f1_csv_path}")
    
    # Save overall metrics to CSV
    overall_data = []
    for epoch in epochs:
        overall = results_per_epoch[epoch]['overall']
        overall_data.append({
            'Epoch': epoch,
            'Micro_F1': overall['micro_f1'],
            'Macro_F1': overall['macro_f1'],
            'Micro_Precision': overall['micro_precision'],
            'Macro_Precision': overall['macro_precision'],
            'Micro_Recall': overall['micro_recall'],
            'Macro_Recall': overall['macro_recall']
        })
    
    overall_df = pd.DataFrame(overall_data)
    overall_csv_path = os.path.join(output_dir, 'overall_metrics_per_epoch.csv')
    overall_df.to_csv(overall_csv_path, index=False)
    print(f"✅ Overall metrics CSV saved: {overall_csv_path}")

def save_detailed_report(results_per_epoch, output_dir):
    """Save detailed report ke file txt"""
    
    report_path = os.path.join(output_dir, 'detailed_validation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("INDOBERT NER MODEL - DETAILED VALIDATION REPORT PER EPOCH\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Epochs: {len(results_per_epoch)}\n")
        f.write("="*80 + "\n\n")
        
        # Overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("-"*50 + "\n")
        
        epochs = list(results_per_epoch.keys())
        label_names = results_per_epoch[epochs[0]]['label_names']
        
        f.write(f"{'Epoch':<8} {'Micro F1':<12} {'Macro F1':<12} {'Micro Prec':<12} {'Macro Prec':<12} {'Micro Rec':<12} {'Macro Rec':<12}\n")
        f.write("-"*80 + "\n")
        
        for epoch in epochs:
            overall = results_per_epoch[epoch]['overall']
            f.write(f"{epoch:<8} {overall['micro_f1']:<12.4f} {overall['macro_f1']:<12.4f} "
                   f"{overall['micro_precision']:<12.4f} {overall['macro_precision']:<12.4f} "
                   f"{overall['micro_recall']:<12.4f} {overall['macro_recall']:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Per label detailed results
        f.write("DETAILED RESULTS PER LABEL\n")
        f.write("-"*50 + "\n")
        
        for label_idx, label_name in enumerate(label_names):
            f.write(f"\n{label_name.upper()}\n")
            f.write("-"*30 + "\n")
            f.write(f"{'Epoch':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n")
            f.write("-"*56 + "\n")
            
            for epoch in epochs:
                per_label = results_per_epoch[epoch]['per_label']
                precision = per_label['precision'][label_idx]
                recall = per_label['recall'][label_idx]
                f1 = per_label['f1'][label_idx]
                support = per_label['support'][label_idx]
                
                f.write(f"{epoch:<8} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<12.0f}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Best performing labels
        f.write("BEST PERFORMING LABELS (Final Epoch)\n")
        f.write("-"*50 + "\n")
        
        final_epoch = max(epochs)
        final_results = results_per_epoch[final_epoch]['per_label']
        
        # Sort by F1 score
        label_f1_pairs = [(label_names[i], final_results['f1'][i]) for i in range(len(label_names))]
        label_f1_pairs.sort(key=lambda x: x[1], reverse=True)
        
        f.write(f"{'Rank':<6} {'Label':<20} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}\n")
        f.write("-"*62 + "\n")
        
        for rank, (label, f1_score) in enumerate(label_f1_pairs, 1):
            label_idx = label_names.index(label)
            precision = final_results['precision'][label_idx]
            recall = final_results['recall'][label_idx]
            
            f.write(f"{rank:<6} {label:<20} {f1_score:<12.4f} {precision:<12.4f} {recall:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Detailed report saved: {report_path}")

def main():
    """Main function"""
    
    print("="*80)
    print("INDOBERT NER MODEL - VALIDATION PER EPOCH")
    print("="*80)
    
    # Configuration
    config = {
        'data_path': '../final_dataset.csv',
        'checkpoint_dir': './outputs_indobert_multilabel',
        'output_dir': './validation_results_per_epoch',
        'model_name': 'indobenchmark/indobert-base-p1',
        'max_length': 512,
        'batch_size': 4,  # Increased for faster evaluation
        'device': 'cpu'
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load tokenizer
    print(f"\nLoading IndoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    print(f"✅ Tokenizer loaded successfully")
    
    # Load and preprocess data
    encodings, labels, label2id, id2label = load_and_preprocess_data_multilabel(
        config['data_path'], tokenizer, config['max_length']
    )
    
    # Use validation split (same as training)
    split_idx = int(len(labels) * 0.8)  # 80% train, 20% validation
    
    val_encodings = {
        'input_ids': encodings['input_ids'][split_idx:],
        'attention_mask': encodings['attention_mask'][split_idx:]
    }
    val_labels = labels[split_idx:]
    
    print(f"\nValidation samples: {len(val_labels)}")
    
    # Create validation dataset
    val_dataset = NERDatasetMultiLabel(val_encodings, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Find all checkpoint files
    checkpoint_files = []
    if os.path.exists(config['checkpoint_dir']):
        for file in os.listdir(config['checkpoint_dir']):
            if file.startswith('checkpoint_epoch_') and file.endswith('.pt'):
                epoch_num = int(file.split('_')[2].split('.')[0])
                checkpoint_files.append((epoch_num, file))
    
    checkpoint_files.sort(key=lambda x: x[0])
    
    if not checkpoint_files:
        print("❌ No checkpoint files found!")
        return
    
    print(f"\nFound {len(checkpoint_files)} checkpoint files:")
    for epoch_num, filename in checkpoint_files:
        print(f"  Epoch {epoch_num}: {filename}")
    
    # Create model
    print(f"\nCreating IndoBERT Multi-Label model...")
    model = IndoBERTNERModelMultiLabel(config['model_name'], len(label2id))
    print(f"✅ Model created successfully")
    
    # Evaluate each checkpoint
    results_per_epoch = {}
    
    for epoch_num, filename in checkpoint_files:
        print(f"\n{'='*50}")
        print(f"Evaluating Epoch {epoch_num}")
        print(f"{'='*50}")
        
        # Load checkpoint
        checkpoint_path = os.path.join(config['checkpoint_dir'], filename)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Evaluate
        results = evaluate_model(model, val_loader, device, id2label)
        results_per_epoch[epoch_num] = results
        
        # Print summary
        overall = results['overall']
        print(f"Micro F1: {overall['micro_f1']:.4f}")
        print(f"Macro F1: {overall['macro_f1']:.4f}")
        print(f"Micro Precision: {overall['micro_precision']:.4f}")
        print(f"Macro Precision: {overall['macro_precision']:.4f}")
        print(f"Micro Recall: {overall['micro_recall']:.4f}")
        print(f"Macro Recall: {overall['macro_recall']:.4f}")
    
    # Create visualizations
    print(f"\n{'='*50}")
    print("Creating Visualizations")
    print(f"{'='*50}")
    create_visualizations(results_per_epoch, config['output_dir'])
    
    # Save detailed report
    print(f"\n{'='*50}")
    print("Saving Detailed Report")
    print(f"{'='*50}")
    save_detailed_report(results_per_epoch, config['output_dir'])
    
    print(f"\n✅ Validation per epoch completed!")
    print(f"Results saved in: {config['output_dir']}")

if __name__ == "__main__":
    main()
