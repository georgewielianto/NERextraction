import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from datetime import datetime
import re
from typing import List, Dict, Tuple

class NERDatasetMultiLabel(Dataset):
    """Dataset dengan pre-computed token labels untuk consistency"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Semua sudah pre-computed, tinggal ambil
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
        return item

class IndoBERTNERModelMultiLabel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        # Load IndoBERT
        try:
            self.bert = AutoModel.from_pretrained(model_name, use_safetensors=True)
        except:
            try:
                self.bert = AutoModel.from_pretrained(model_name, use_safetensors=False)
            except:
                self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        self.dropout = nn.Dropout(0.1)
        # Multi-label classifier (no softmax, use sigmoid)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Forward pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # BCEWithLogitsLoss untuk multi-label classification
            loss_fct = nn.BCEWithLogitsLoss()
            
            # Only calculate loss on non-padded tokens
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1, labels.size(-1))[active_loss]
            
            loss = loss_fct(active_logits, active_labels)
        
        return {'loss': loss, 'logits': logits}

class IndoBERTTrainerMultiLabel:
    def __init__(self, model, train_loader, val_loader, device, config, id2label, resume_path=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.id2label = id2label
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * config['num_epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        
        # Best model tracking
        self.best_f1 = 0.0
        self.best_model_path = None
        
        # Resume from checkpoint if provided
        self.start_epoch = 0
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint untuk resume training"""
        print(f"🔄 Loading checkpoint from: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training history
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.val_f1_scores = checkpoint.get('val_f1_scores', [])
            
            # Load best F1 score
            self.best_f1 = checkpoint.get('f1_score', 0.0)
            
            # Set start epoch
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            
            print(f"✅ Checkpoint loaded successfully!")
            print(f"   Resume from epoch: {self.start_epoch + 1}")
            print(f"   Previous best F1: {self.best_f1:.4f}")
            print(f"   Training history: {len(self.train_losses)} epochs")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            self.start_epoch = 0
    
    def train_epoch(self, epoch):
        """Train satu epoch"""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate dengan multi-label metrics"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                total_loss += loss.item()
                
                # Get predictions (sigmoid + threshold)
                logits = outputs['logits']
                predictions = torch.sigmoid(logits) > 0.5
                
                # Collect predictions and labels (only non-padded tokens)
                for i in range(len(predictions)):
                    seq_len = attention_mask[i].sum().item()
                    all_predictions.extend(predictions[i][:seq_len].cpu().numpy())
                    all_labels.extend(labels[i][:seq_len].cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        # Calculate multi-label F1 score
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Micro F1 (overall)
        f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
        
        # Macro F1 (per label)
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        # Weighted F1
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        print(f"\n  Micro F1: {f1_micro:.4f}")
        print(f"  Macro F1: {f1_macro:.4f}")
        print(f"  Weighted F1: {f1_weighted:.4f}")
        
        self.val_f1_scores.append(f1_micro)
        
        return avg_loss, f1_micro
    
    def save_model(self, epoch, f1_score, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'config': self.config,
            'f1_score': f1_score,
            'model_type': 'IndoBERT-MultiLabel',
            'id2label': self.id2label
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['output_dir'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            print(f"✅ New best model saved with F1: {f1_score:.4f}")
    
    def train(self):
        """Main training loop"""
        if self.start_epoch > 0:
            print(f"🔄 Resuming IndoBERT Multi-Label training from epoch {self.start_epoch + 1}...")
        else:
            print(f"Starting IndoBERT Multi-Label training for {self.config['num_epochs']} epochs...")
        
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_f1 = self.validate_epoch(epoch)
            
            # Print results
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val F1 Score (Micro): {val_f1:.4f}")
            
            # Save model
            is_best = val_f1 > self.best_f1
            if is_best:
                self.best_f1 = val_f1
            
            self.save_model(epoch, val_f1, is_best)
            
            # Early stopping - DISABLED
            if self.config.get('early_stopping_patience') is not None:
                if epoch >= self.config['early_stopping_patience']:
                    recent_f1s = self.val_f1_scores[-self.config['early_stopping_patience']:]
                    if val_f1 < max(recent_f1s):
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
            else:
                print(f"🚫 Early stopping disabled - training akan berlanjut sampai epoch {self.config['num_epochs']}")
        
        print(f"\n✅ Training completed!")
        print(f"Best F1 Score: {self.best_f1:.4f}")
        print(f"Best model saved at: {self.best_model_path}")

def encode_labels_with_alignment(texts, annotations_list, tokenizer, label2id, max_length):
    """
    Encode texts dan align labels SEKALI di preprocessing
    Mengembalikan encodings dan aligned labels yang sudah siap pakai
    """
    num_labels = len(label2id)
    
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    skipped_samples = 0
    alignment_errors = 0
    overlapping_count = 0
    
    print(f"\nEncoding {len(texts)} samples with alignment...")
    
    for idx, (text, annotations) in enumerate(zip(texts, annotations_list)):
        # Tokenize dengan offset_mapping SEKALI
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
            skipped_samples += 1
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
                    alignment_errors += 1
                    continue
                
                # Mark all entity types for this span //bagian multihot encoding
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
                
                # Count overlapping
                if np.sum(token_label) > 1:
                    overlapping_count += 1
            
            token_labels.append(token_label.tolist())
        
        # Simpan encodings dan labels
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(token_labels)
        
        if (idx + 1) % 100 == 0:
            print(f"Encoded {idx + 1}/{len(texts)} samples (skipped: {skipped_samples}, errors: {alignment_errors})")
    
    print(f"\n✅ Encoding completed!")
    print(f"   Valid samples: {len(all_input_ids)}")
    print(f"   Skipped samples: {skipped_samples}")
    print(f"   Alignment errors: {alignment_errors}")
    print(f"   Tokens with overlapping entities: {overlapping_count}")
    
    # Return as dict untuk consistency
    encodings = {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks
    }
    
    return encodings, all_labels

def load_and_preprocess_data_multilabel(data_path: str, tokenizer, max_length: int = 512):
    """Load dan preprocess data dengan multi-label encoding untuk overlapping entities"""
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
        
        if (idx + 1) % 100 == 0:
            print(f"Parsed {idx + 1}/{len(df)} rows")
    
    print(f"\n✅ Parsed {len(texts)} valid texts")
    
    # Encode dengan alignment SEKALI di sini
    encodings, labels = encode_labels_with_alignment(
        texts, annotations_list, tokenizer, label2id, max_length
    )
    
    # Print statistics
    if labels:
        total_labels = sum(sum(sum(label) for label in sample) for sample in labels)
        avg_labels_per_sample = total_labels / len(labels)
        print(f"Average labels per sample: {avg_labels_per_sample:.2f}")
    
    return encodings, labels, label2id, id2label

def main():
    """Main function"""
    
    print("="*80)
    print("TRAINING INDOBERT NER MODEL - MULTI-LABEL (OVERLAPPING ENTITIES)")
    print("="*80)
    
    # Configuration
    config = {
        'data_path': '../final_dataset.csv',
        'output_dir': './outputs_indobert_multilabel',
        'model_name': 'indobenchmark/indobert-base-p1',
        'max_length': 512,
        'batch_size': 2,
        'learning_rate': 2e-5,
        'num_epochs': 15,
        'weight_decay': 0.01,
        'train_split': 0.8,
        'early_stopping_patience': None,  # DISABLED: Set ke None untuk disable early stopping
        'device': 'cpu'
    }
    
    # Check for resume checkpoint
    resume_path = None
    checkpoint_dir = config['output_dir']
    
    # Cari checkpoint terbaru jika ada
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
        if checkpoint_files:
            # Ambil checkpoint dengan epoch tertinggi
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
            resume_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"🔄 Found checkpoint: {latest_checkpoint}")
            print(f"   Resume path: {resume_path}")
    
    if resume_path:
        print(f"📁 Resume training from: {resume_path}")
    else:
        print("🆕 Starting fresh training...")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save configuration
    config['timestamp'] = datetime.now().isoformat()
    config_path = os.path.join(config['output_dir'], 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = str(device)
    
    # Load tokenizer
    print(f"\nLoading IndoBERT tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        print(f"✅ IndoBERT tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return
    
    # Load and preprocess data - ENCODING SEKALI DI SINI
    encodings, labels, label2id, id2label = load_and_preprocess_data_multilabel(
        config['data_path'], tokenizer, config['max_length']
    )
    
    if len(labels) == 0:
        print("❌ No valid samples found. Please check your dataset.")
        return
    
    # Split data
    split_idx = int(len(labels) * config['train_split'])
    
    train_encodings = {
        'input_ids': encodings['input_ids'][:split_idx],
        'attention_mask': encodings['attention_mask'][:split_idx]
    }
    train_labels = labels[:split_idx]
    
    val_encodings = {
        'input_ids': encodings['input_ids'][split_idx:],
        'attention_mask': encodings['attention_mask'][split_idx:]
    }
    val_labels = labels[split_idx:]
    
    print(f"\nTrain samples: {len(train_labels)}")
    print(f"Val samples: {len(val_labels)}")
    
    # Create datasets - pass pre-computed encodings
    train_dataset = NERDatasetMultiLabel(train_encodings, train_labels)
    val_dataset = NERDatasetMultiLabel(val_encodings, val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    print(f"\nCreating IndoBERT Multi-Label model...")
    try:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        model = IndoBERTNERModelMultiLabel(config['model_name'], len(label2id))
        print(f"✅ IndoBERT Multi-Label model created successfully")
        print(f"   Model output dimension: {len(label2id)} (multi-label)")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # Create trainer dengan resume path
    trainer = IndoBERTTrainerMultiLabel(model, train_loader, val_loader, device, config, id2label, resume_path)
    
    # Start training
    trainer.train()
    
    print(f"\n✅ Training completed!")
    print(f"Best model saved at: {trainer.best_model_path}")

if __name__ == "__main__":
    main()