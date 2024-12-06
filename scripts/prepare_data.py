import os
import json
import pandas as pd
from datasets import load_dataset
from typing import Dict, List
from tqdm import tqdm
import logging
import argparse
import shutil
import re
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define content filtering patterns
FILTER_PATTERNS = {
    'crisis': [
        r'suicid\w*',
        r'kill\w*',
        r'harm\w*',
        r'hurt\w*',
        r'die\w*',
        r'death\w*'
    ],
    'inappropriate': [
        r'sex\w*',
        r'nsfw',
        r'explicit',
        r'drug\w*',
        r'abuse\w*'
    ]
}

def clean_directory(base_dir: str):
    """Remove old directories and create a clean structure."""
    # Remove the entire raw directory if it exists
    raw_dir = os.path.join(base_dir, 'raw')
    if os.path.exists(raw_dir):
        logger.info(f"Removing old raw directory: {raw_dir}")
        shutil.rmtree(raw_dir)
    
    # Remove processed directory if it exists
    processed_dir = os.path.join(base_dir, 'processed')
    if os.path.exists(processed_dir):
        logger.info(f"Removing old processed directory: {processed_dir}")
        shutil.rmtree(processed_dir)

def setup_directories(base_dir: str) -> Dict[str, str]:
    """Create necessary directories for data processing."""
    # Clean old directories first
    clean_directory(base_dir)
    
    # Create new directory structure
    dirs = {
        'raw': os.path.join(base_dir, 'raw'),
        'processed': os.path.join(base_dir, 'processed'),
        'mental_health': os.path.join(base_dir, 'raw', 'mental_health'),
        'therapy': os.path.join(base_dir, 'raw', 'therapy')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
        
    return dirs

def filter_text(text: str) -> bool:
    """Return True if text passes all filters."""
    text_lower = text.lower()
    
    # Check against all filter patterns
    for category, patterns in FILTER_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return False
    
    return True

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters except punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
    return text.strip()

def download_mental_health_dataset(output_dir: str) -> pd.DataFrame:
    """Download and save the Counsel Chat dataset."""
    logger.info("Downloading Counsel Chat dataset...")
    
    try:
        dataset = load_dataset("nbertagnolli/counsel-chat")
        logger.info(f"Downloaded dataset with {len(dataset['train'])} examples")
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'user': [x['questionText'] for x in dataset['train']],
            'assistant': [x['answerText'] for x in dataset['train']]
        })
        
        # Clean and filter data
        initial_len = len(df)
        logger.info(f"Initial examples: {initial_len}")
        
        # Remove missing values
        df = df.dropna()
        logger.info(f"After removing missing values: {len(df)}")
        
        # Clean text
        df['user'] = df['user'].apply(clean_text)
        df['assistant'] = df['assistant'].apply(clean_text)
        
        # Apply length filters
        df = df[df['user'].str.len().between(10, 500)]
        df = df[df['assistant'].str.len().between(10, 1000)]
        logger.info(f"After length filtering: {len(df)}")
        
        # Apply content filters
        df = df[df['user'].apply(filter_text) & df['assistant'].apply(filter_text)]
        logger.info(f"After content filtering: {len(df)}")
        
        # Save processed data
        output_file = os.path.join(output_dir, 'counsel_chat.csv')
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} cleaned examples to {output_file}")
        logger.info(f"Removed {initial_len - len(df)} examples during cleaning")
        
        return df
    except Exception as e:
        logger.error(f"Error downloading Counsel Chat dataset: {str(e)}")
        raise

def download_therapy_dataset(output_dir: str) -> pd.DataFrame:
    """Download and save the therapy dataset."""
    logger.info("Downloading Prosocial Dialog dataset...")
    
    try:
        dataset = load_dataset("allenai/prosocial-dialog", split='train')
        logger.info(f"Downloaded dataset with {len(dataset)} examples")
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'user': dataset['context'],
            'assistant': dataset['response']
        })
        
        # Clean and filter data
        initial_len = len(df)
        logger.info(f"Initial examples: {initial_len}")
        
        # Remove missing values
        df = df.dropna()
        logger.info(f"After removing missing values: {len(df)}")
        
        # Clean text
        df['user'] = df['user'].apply(clean_text)
        df['assistant'] = df['assistant'].apply(clean_text)
        
        # Apply length filters
        df = df[df['user'].str.len().between(10, 500)]
        df = df[df['assistant'].str.len().between(10, 1000)]
        logger.info(f"After length filtering: {len(df)}")
        
        # Apply content filters
        df = df[df['user'].apply(filter_text) & df['assistant'].apply(filter_text)]
        logger.info(f"After content filtering: {len(df)}")
        
        # Save processed data
        output_file = os.path.join(output_dir, 'therapy_conversations.csv')
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} cleaned examples to {output_file}")
        logger.info(f"Removed {initial_len - len(df)} examples during cleaning")
        
        return df
    except Exception as e:
        logger.error(f"Error downloading Prosocial Dialog dataset: {str(e)}")
        raise

def balance_datasets(counsel_data: List[Dict], therapy_data: List[Dict], max_ratio: float = 3.0) -> List[Dict]:
    """Balance datasets to prevent domination by any one source."""
    if not counsel_data or not therapy_data:
        return counsel_data + therapy_data
    
    # Get lengths
    len_counsel = len(counsel_data)
    len_therapy = len(therapy_data)
    
    # Calculate ratio
    ratio = max(len_counsel / len_therapy, len_therapy / len_counsel)
    
    # Balance if ratio exceeds max_ratio
    if ratio > max_ratio:
        if len_counsel > len_therapy:
            target_len = int(len_therapy * max_ratio)
            random.shuffle(counsel_data)
            counsel_data = counsel_data[:target_len]
        else:
            target_len = int(len_counsel * max_ratio)
            random.shuffle(therapy_data)
            therapy_data = therapy_data[:target_len]
    
    return counsel_data + therapy_data

def process_datasets(dirs: Dict[str, str], output_dir: str):
    """Process and combine all datasets into a unified format."""
    logger.info("Processing datasets...")
    
    try:
        # Process Counsel Chat
        counsel_file = os.path.join(dirs['mental_health'], 'counsel_chat.csv')
        if not os.path.exists(counsel_file):
            logger.error(f"Counsel Chat data file not found: {counsel_file}")
            raise FileNotFoundError(f"Missing file: {counsel_file}")
            
        counsel_df = pd.read_csv(counsel_file)
        logger.info(f"Loaded {len(counsel_df)} examples from Counsel Chat")
        
        counsel_processed = []
        for _, row in tqdm(counsel_df.iterrows(), desc="Processing Counsel Chat dataset", total=len(counsel_df)):
            counsel_processed.append({
                'instruction': row['user'],
                'response': row['assistant'],
                'source': 'counsel_chat',
                'category': 'mental_health'
            })
        
        # Process Therapy Conversations
        therapy_file = os.path.join(dirs['therapy'], 'therapy_conversations.csv')
        if not os.path.exists(therapy_file):
            logger.error(f"Therapy conversations file not found: {therapy_file}")
            raise FileNotFoundError(f"Missing file: {therapy_file}")
            
        therapy_df = pd.read_csv(therapy_file)
        logger.info(f"Loaded {len(therapy_df)} examples from Therapy conversations")
        
        therapy_processed = []
        for _, row in tqdm(therapy_df.iterrows(), desc="Processing Therapy dataset", total=len(therapy_df)):
            therapy_processed.append({
                'instruction': row['user'],
                'response': row['assistant'],
                'source': 'prosocial_dialog',
                'category': 'therapy'
            })
        
        # Balance datasets
        logger.info("Balancing datasets...")
        all_data = balance_datasets(counsel_processed, therapy_processed)
        logger.info(f"Total processed examples after balancing: {len(all_data)}")
        
        # Save processed data
        output_file = os.path.join(output_dir, 'processed_data.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        # Log statistics
        counsel_count = sum(1 for x in all_data if x['source'] == 'counsel_chat')
        therapy_count = sum(1 for x in all_data if x['source'] == 'prosocial_dialog')
        
        logger.info(f"Dataset statistics after balancing:")
        logger.info(f"- Counsel Chat examples: {counsel_count}")
        logger.info(f"- Therapy examples: {therapy_count}")
        logger.info(f"- Total examples: {len(all_data)}")
        logger.info(f"- Ratio (larger/smaller): {max(counsel_count/therapy_count, therapy_count/counsel_count):.2f}")
        logger.info(f"Saved processed data to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing datasets: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Download and process therapy datasets')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Base directory for datasets')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--max_ratio', type=float, default=3.0, help='Maximum ratio between dataset sizes')
    args = parser.parse_args()
    
    try:
        # Setup directories
        dirs = setup_directories(args.dataset_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Download datasets
        download_mental_health_dataset(dirs['mental_health'])
        download_therapy_dataset(dirs['therapy'])
        
        # Process datasets
        process_datasets(dirs, args.output_dir)
        
    except Exception as e:
        logger.error(f"Failed to prepare data: {str(e)}")
        raise

if __name__ == "__main__":
    main() 