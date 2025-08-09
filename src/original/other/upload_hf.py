#!/opt/homebrew/bin/python3.12
from huggingface_hub import HfApi

def main():
    # Initialize the API client.
    api = HfApi()
    
    # Use upload_large_folder for a resilient upload of a large folder.
    # This method splits the upload into multiple commits, making it more robust.
    api.upload_large_folder(
        repo_id="VerisimilitudeX/EpiMECoV",
        repo_type="dataset",
        folder_path="/Volumes/T9/EpiMECoV1"
    )

if __name__ == "__main__":
    main()