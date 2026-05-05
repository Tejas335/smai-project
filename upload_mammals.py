import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

def upload_to_huggingface():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the HF API key
    hf_token = os.getenv("HF_API_KEY")
    if not hf_token:
        raise ValueError("HF_API_KEY not found in .env file")
        
    api = HfApi(token=hf_token)
    repo_id = "ViratGarg/animal_species_SMAI"
    local_dir = "dataset/mammals"
    
    print(f"Uploading {local_dir} to https://huggingface.co/datasets/{repo_id}...")
    
    # Upload the entire directory preserving its structure
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="mammals"
    )
    
    print("Upload completed successfully!")

if __name__ == "__main__":
    upload_to_huggingface()
