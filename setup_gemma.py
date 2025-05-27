#!/usr/bin/env python3

import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import whoami, login
import torch

def check_authentication():
    """Check if user is authenticated with Hugging Face."""
    try:
        user_info = whoami()
        return True, user_info['name']
    except Exception:
        return False, None

def setup_gemma():
    """Download and cache Gemma-2B model for first-time use."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    model_name = "google/gemma-2b"
    
    # Check authentication first
    logger.info("Checking Hugging Face authentication...")
    is_authenticated, username = check_authentication()
    
    if not is_authenticated:
        logger.error("❌ Not authenticated with Hugging Face!")
        logger.error("Please authenticate using one of these methods:")
        logger.error("1. Run: huggingface-cli login")
        logger.error("2. Set HF_TOKEN environment variable")
        logger.error("3. Add HF_TOKEN to your .env file")
        logger.error("\nGet your token from: https://huggingface.co/settings/tokens")
        return False
    else:
        logger.info(f"✓ Authenticated as: {username}")
    
    try:
        logger.info(f"Downloading Gemma-2B model: {model_name}")
        logger.info("This may take a few minutes on first run...")
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("✓ Tokenizer downloaded successfully")
        
        # Download model
        logger.info("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        logger.info("✓ Model downloaded successfully")
        
        # Test the model
        logger.info("Testing model...")
        test_input = "This is a test log: INFO Application started"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        logger.info("✓ Model test completed successfully")
        logger.info("Gemma-2B is ready to use!")
        
        # Print system info
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to set up Gemma-2B: {str(e)}")
        
        if "gated repo" in str(e) or "restricted" in str(e):
            logger.error("\n🔒 This appears to be an access issue:")
            logger.error("1. Make sure you've accepted the Gemma license at:")
            logger.error("   https://huggingface.co/google/gemma-2b")
            logger.error("2. Ensure you're authenticated (run: huggingface-cli whoami)")
            logger.error("3. Try logging out and back in: huggingface-cli logout && huggingface-cli login")
        else:
            logger.error("Make sure you have:")
            logger.error("1. Sufficient memory/storage space")
            logger.error("2. Internet connection for model download")
            logger.error("3. Proper PyTorch installation")
        return False

if __name__ == "__main__":
    success = setup_gemma()
    if success:
        print("\n🎉 Setup complete! You can now run the log analyzer with Gemma-2B.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.") 