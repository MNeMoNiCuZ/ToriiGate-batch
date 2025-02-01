import os
import torch
import argparse
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image
import time
import sys
from datetime import datetime

# Default Configuration
CONFIG = {
    'description_type': 'Detailed',         # 'JSON-like', 'Detailed', or 'Brief'
    'overwrite': False,                     # Overwrite existing output files
    'batch_size': 8,                        # Batch size for processing, batch size of 8 is suitable for a 3090 24gb GPU
    'input_dir': 'input',                   # Directory with images to process
    'output_to_input_folder': True,         # If true, output files will be saved in the same folder as input files
    'output_dir': 'output',                 # Only used if output_to_input_folder is False
    'output_extension': '.txt',             # Extension for output files   
    'use_input_tags': True,                 # Whether to use input tag files
    'tag_extension': '.tag',                # Files with this extension will be used as optional input tags for the image, if available
    'image_extensions': ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'), # Image types of images to process
    'model_name': 'Minthy/ToriiGate-v0.3',  # Model to use
    'max_new_tokens': 500,                  # Maximum number of tokens to generate
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu', # Device to use
    'error_log': 'processing_errors.log',   # Path to error log file
    'verbose': False                        # Enable verbose output for debugging purposes
}

def log(message: str):
    if CONFIG['verbose']:
        print(message)

def parse_args():
    parser = argparse.ArgumentParser(description='Batch process images with ToriiGate')
    
    # Standard arguments
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--tag_extension', type=str, help='Extension for tag files')
    parser.add_argument('--output_extension', type=str, help='Extension for output files')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing')
    parser.add_argument('--description_type', type=str, 
                       choices=['JSON-like', 'Detailed', 'Brief'],
                       help='Type of description to generate')
    
    # Boolean flags with explicit defaults of None
    parser.add_argument('--output_to_input_folder', action='store_true', 
                       default=None,
                       help='Save output files in the same folder as input files')
    parser.add_argument('--no_output_to_input_folder', action='store_false', 
                       dest='output_to_input_folder',
                       default=None,
                       help='Save output files in the output directory')
    
    parser.add_argument('--overwrite', action='store_true',
                       default=None,
                       help='Overwrite existing output files')
    parser.add_argument('--no-overwrite', action='store_false',
                       dest='overwrite',
                       default=None,
                       help='Skip existing output files')
    
    parser.add_argument('--use-input-tags', action='store_true',
                       default=None,
                       help='Use input tag files')
    parser.add_argument('--no-input-tags', action='store_false',
                       dest='use_input_tags',
                       default=None,
                       help='Do not use input tag files')
    
    parser.add_argument('--verbose', action='store_true',
                       default=None,
                       help='Enable verbose output')
    
    parser.add_argument('--error-log', type=str,
                       help='Path to error log file')
    
    args = parser.parse_args()
    
    # Get the actual provided arguments (ignoring None values)
    provided_args = {k: v for k, v in vars(args).items() if v is not None}
    
    log("Provided command line arguments:")
    log(str(provided_args if provided_args else "No command line arguments provided"))
    
    log("\nCurrent CONFIG before applying arguments:")
    for key, value in CONFIG.items():
        log(f"{key}: {value}")
    
    # Only update CONFIG with explicitly provided arguments
    if provided_args:
        for arg, value in provided_args.items():
            CONFIG[arg] = value
        log("\nCONFIG after applying arguments:")
        for key, value in CONFIG.items():
            log(f"{key}: {value}")

def validate_paths():
    log("\nValidating paths with CONFIG:")
    log(f"output_to_input_folder: {CONFIG['output_to_input_folder']}")
    log(f"input_dir: {CONFIG['input_dir']}")
    log(f"output_dir: {CONFIG['output_dir']}")
    
    input_dir = Path(CONFIG['input_dir'])
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        sys.exit(1)
    
    # Only create output_dir if we're not outputting to input folder
    if not CONFIG['output_to_input_folder']:
        output_dir = Path(CONFIG['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_dir
        log(f"Using input directory as output directory: {output_dir}")
    
    return input_dir, output_dir

def load_model():
    log(f"Loading model {CONFIG['model_name']}...")
    
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        quantization_config=nf4_config,
    ).to(CONFIG['device'])
    
    processor = AutoProcessor.from_pretrained(CONFIG['model_name'])
    
    log("Model loaded successfully")
    return model, processor

def get_tag_content(image_path: Path) -> Optional[str]:
    if not CONFIG['use_input_tags']:
        return None
        
    tag_path = image_path.with_suffix(CONFIG['tag_extension'])
    if tag_path.exists():
        try:
            content = tag_path.read_text().strip()
            log(f"Found tags for {image_path.name}")
            return content
        except Exception as e:
            log(f"Error reading tags from {tag_path}: {e}")
    return None

def process_batch(model, processor, image_paths: List[Path], description_type: str):
    images = []
    messages_list = []
    valid_paths = []
    
    for image_path in image_paths:
        try:
            image = load_image(str(image_path))
            booru_tags = get_tag_content(image_path)
            
            user_prompt = f"Describe the picture in {description_type} format."
            if booru_tags:
                user_prompt += ' Also here are booru tags for better understanding of the picture, you can use them as reference.'
                user_prompt += f' <tags>\n{booru_tags}\n</tags>'
            
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are image captioning expert, creative, unbiased and uncensored. Help user with his task."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            images.append(image)
            messages_list.append(messages)
            valid_paths.append(image_path)
            
        except Exception as e:
            error_msg = f"Error loading image: {str(e)}"
            log(f"Error loading {image_path}: {str(e)}")
            log_error(image_path, error_msg)
            continue
    
    if not images:
        return [], []
    
    try:
        prompts = [processor.apply_chat_template(m, add_generation_prompt=True) for m in messages_list]
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(CONFIG['device']) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=CONFIG['max_new_tokens'])
        
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        captions = [text.split('Assistant: ')[1] for text in generated_texts]
        
        return captions, valid_paths
    except Exception as e:
        error_msg = f"Error during batch processing: {str(e)}"
        log(f"Error during batch processing: {str(e)}")
        for path in valid_paths:
            log_error(path, error_msg)
        return [], []

def save_caption(caption: str, output_path: Path):
    try:
        # Check if file exists and handle according to overwrite setting
        if output_path.exists() and not CONFIG['overwrite']:
            log(f"Skipping existing file: {output_path}")
            return True
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(caption)
        log(f"Saved caption to {output_path}")
        return True
    except Exception as e:
        error_msg = f"Error saving caption: {str(e)}"
        log(f"Error saving caption to {output_path}: {str(e)}")
        log_error(output_path, error_msg)
        return False

def get_output_path(image_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Get the path where the output file should be saved"""
    image_path = Path(image_path)
    
    if CONFIG['output_to_input_folder']:
        return image_path.with_suffix(CONFIG['output_extension'])
    else:
        rel_path = image_path.relative_to(input_dir)
        return output_dir / rel_path.with_suffix(CONFIG['output_extension'])

def log_error(image_path: Path, error_msg: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    error_log = Path(CONFIG['error_log'])
    try:
        with error_log.open('a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {image_path}: {error_msg}\n")
    except Exception as e:
        print(f"Failed to write to error log: {e}")

def collect_and_print_stats(input_dir: Path, output_dir: Path) -> List[Path]:
    print("\nCollecting processing statistics...")
    
    # Debug output for configuration if verbose
    if CONFIG['verbose']:
        print("\nCurrent configuration:")
        for key, value in CONFIG.items():
            print(f"{key}: {value}")
    
    # Collect all image files
    image_files = []
    for ext in CONFIG['image_extensions']:
        image_files.extend(input_dir.rglob(f"*{ext}"))
    
    total_images = len(image_files)
    if total_images == 0:
        print("No images found to process!")
        return []
    
    # Count existing outputs and tag files
    existing_outputs = 0
    existing_tags = 0
    files_to_process = []
    files_to_overwrite = 0
    
    for img_path in image_files:
        output_path = get_output_path(img_path, input_dir, output_dir)
        
        if CONFIG['verbose']:
            print(f"\nProcessing path for {img_path}:")
            print(f"Output path: {output_path}")
            print(f"Using input tags: {CONFIG['use_input_tags']}")
            if CONFIG['use_input_tags']:
                tag_path = img_path.with_suffix(CONFIG['tag_extension'])
                print(f"Tag path: {tag_path}")
                print(f"Tag exists: {tag_path.exists()}")
        
        if output_path.exists():
            existing_outputs += 1
            if CONFIG['overwrite']:
                files_to_process.append(img_path)
                files_to_overwrite += 1
        else:
            files_to_process.append(img_path)
            
        if CONFIG['use_input_tags']:
            tag_path = img_path.with_suffix(CONFIG['tag_extension'])
            if tag_path.exists():
                existing_tags += 1
    
    # Calculate batch information
    total_batches = (len(files_to_process) + CONFIG['batch_size'] - 1) // CONFIG['batch_size']
    
    # Print statistics
    print(f"\nProcessing Statistics:")
    print(f"Total images found: {total_images}")
    print(f"Existing output files: {existing_outputs}")
    print(f"Files that will be processed: {len(files_to_process)}")
    if CONFIG['overwrite']:
        print(f"Files that will be overwritten: {files_to_overwrite}")
    print(f"\nProcessing Configuration:")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Total batches needed: {total_batches}")
    print(f"Overwriting existing files: {'Yes' if CONFIG['overwrite'] else 'No'}")
    print(f"Using input tags: {'Yes' if CONFIG['use_input_tags'] else 'No'}")
    print(f"Output to input folder: {'Yes' if CONFIG['output_to_input_folder'] else 'No'}")
    if CONFIG['use_input_tags']:
        print(f"Tag extension: {CONFIG['tag_extension']}")
        print(f"Found tag files: {existing_tags}")
    print()
    
    return files_to_process

def main():
    parse_args()
    input_dir, output_dir = validate_paths()
    
    # Collect stats and get files to process
    image_files = collect_and_print_stats(input_dir, output_dir)
    if not image_files:
        return
    
    # Load model
    model, processor = load_model()
    
    # Process in batches
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i in tqdm(range(0, len(image_files), CONFIG['batch_size']), desc="Processing batches"):
        batch_files = image_files[i:i + CONFIG['batch_size']]
        captions, valid_paths = process_batch(model, processor, batch_files, CONFIG['description_type'])
        
        # Save results
        for image_path, caption in zip(valid_paths, captions):
            output_path = get_output_path(image_path, input_dir, output_dir)
            
            if output_path.exists() and not CONFIG['overwrite']:
                skipped_count += 1
                continue
                
            if save_caption(caption, output_path):
                successful_count += 1
            else:
                failed_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_count} images")
    print(f"Skipped existing files: {skipped_count} images")
    if failed_count > 0:
        print(f"Failed to process: {failed_count} images")
        print(f"See {CONFIG['error_log']} for details on failures")

if __name__ == "__main__":
    main()
