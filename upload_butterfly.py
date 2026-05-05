import os
import shutil
import tempfile
from dotenv import load_dotenv
from huggingface_hub import HfApi

def upload_butterflies():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the HF API key
    hf_token = os.getenv("HF_API_KEY")
    if not hf_token:
        raise ValueError("HF_API_KEY not found in .env file")
        
    api = HfApi(token=hf_token)
    repo_id = "ViratGarg/animal_species_SMAI"
    
    print(f"Deleting existing 'butterfly' folder from repo {repo_id}...")
    try:
        api.delete_folder(path_in_repo="butterfly", repo_id=repo_id, repo_type="dataset")
        print("Successfully deleted existing 'butterfly' folder.")
    except Exception as e:
        print(f"Directory might not exist or couldn't be deleted: {e}")
    
    # Path to the Butterfly dataset
    # Wait, the folder attachment says `dataset/Butterfly` or `dataset/archive-5`?
    # Ah, the user attached `/Users/virat_garg/Documents/Sem 6/SMAI_project/dataset/Butterfly`.
    base_dir = "dataset/Butterfly"
    train_dir = os.path.join(base_dir, "train")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
        
    # Get all class names by listing directories in the 'train' folder
    all_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    # Sort them alphabetically and take the first 30
    all_classes.sort()
    target_classes = all_classes[:30]
    
    print(f"Selected 30 classes: {target_classes}")
    
    # Create a temporary directory to structure the data for upload
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Structuring data in a temporary directory...")
        
        for cls in target_classes:
            target_cls_dir = os.path.join(temp_dir, cls)
            os.makedirs(target_cls_dir, exist_ok=True)
            
            # Combine images from train, valid, and test
            for split in ["train", "valid", "test"]:
                split_cls_dir = os.path.join(base_dir, split, cls)
                if not os.path.exists(split_cls_dir):
                    continue
                    
                for filename in os.listdir(split_cls_dir):
                    src_path = os.path.join(split_cls_dir, filename)
                    if os.path.isfile(src_path):
                        # Use the original filename
                        dst_path = os.path.join(target_cls_dir, filename)
                        
                        # Handle potential filename collisions without using split names
                        counter = 1
                        while os.path.exists(dst_path):
                            name, ext = os.path.splitext(filename)
                            dst_path = os.path.join(target_cls_dir, f"{name}_{counter}{ext}")
                            counter += 1
                        
                        shutil.copy2(src_path, dst_path)
                        
        print(f"Uploading top 30 structured classes to https://huggingface.co/datasets/{repo_id}...")
        
        # Upload the entire temporary directory preserving its structure
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="butterfly"
        )
        
        print("Upload completed successfully!")

if __name__ == "__main__":
    upload_butterflies()
