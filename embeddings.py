import os
import torch
import open_clip
from datasets import load_dataset
from tqdm import tqdm

def extract_and_save_embeddings():
    print("Loading Standard CLIP model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Using the standard OpenAI CLIP model through open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    model = model.to(device)
    model.eval()

    categories = ["mammals", "birds", "butterfly"]
    
    for category in categories:
        print(f"\n--- Processing category: {category} ---")
        print(f"Loading dataset for {category}...")
        # Load dataset from Hugging Face specific to the category directory
        dataset = load_dataset("ViratGarg/animal_species_SMAI", data_dir=category, split="train")
        
        # Split the dataset into train and test
        print(f"Splitting dataset into train and test for {category}...")
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        output_dir = os.path.join("embeddings_clip", category)
        os.makedirs(output_dir, exist_ok=True)
        
        def process_split(split_name, data_split):
            print(f"Processing {split_name} split with {len(data_split)} images...")
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Check if dataset has labels, if not we just save them by index or image path
            has_labels = 'label' in data_split.features
            
            # We can store embeddings systematically in class folders if labels exist
            if has_labels:
                labels = data_split.features['label'].names
                for label in labels:
                    os.makedirs(os.path.join(split_dir, label), exist_ok=True)
            
            batch_size = 32
            for i in tqdm(range(0, len(data_split), batch_size), desc=f"Extracting {split_name}"):
                batch = data_split[i:i+batch_size]
                images = [preprocess(img).unsqueeze(0) for img in batch['image']]
                
                with torch.no_grad():
                    image_tensors = torch.cat(images).to(device)
                    features = model.encode_image(image_tensors)
                    features = features / features.norm(dim=-1, keepdim=True)
                
                for j in range(len(batch['image'])):
                    idx = i + j
                    feat_np = features[j].cpu().numpy()
                    
                    # Determine save path
                    if has_labels:
                        class_name = labels[batch['label'][j]]
                        save_path = os.path.join(split_dir, class_name, f"img_{idx}.pt")
                    else:
                        save_path = os.path.join(split_dir, f"img_{idx}.pt")
                    
                    torch.save(feat_np, save_path)

        process_split("train", dataset["train"])
        process_split("test", dataset["test"])
        
    print("\nAll embeddings extracted and saved successfully.")

if __name__ == "__main__":
    extract_and_save_embeddings()
