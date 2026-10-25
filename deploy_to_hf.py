"""
1-Click Automated Cloud Deployment to Hugging Face Spaces (24/7 Free Hosting)
Run: python deploy_to_hf.py
"""

import os
import sys

def main():
    print("=" * 60)
    print("🎗️ Sanjeevani AI — 1-Click 24/7 Cloud Space Deployment")
    print("=" * 60)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face Access Token (from https://huggingface.co/settings/tokens): ").strip()
    
    if not token:
        print("[ERROR] Hugging Face token is required for deployment.")
        return

    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"[OK] Authenticated as Hugging Face user: {username}")
    except Exception as e:
        print(f"[ERROR] Failed to authenticate token: {e}")
        return

    space_name = input(f"Enter Space Name [default: sanjeevani-ai]: ").strip() or "sanjeevani-ai"
    repo_id = f"{username}/{space_name}"

    print(f"\nCreating / Updating Space '{repo_id}'...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True
        )
        print(f"[OK] Space repository verified: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"[ERROR] Failed creating repo: {e}")
        return

    deploy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_space_deployment")
    print(f"Uploading files from '{deploy_dir}' to Space...")
    try:
        api.upload_folder(
            folder_path=deploy_dir,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy Sanjeevani AI Multimodal Affective & CanSupport Platform"
        )
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Your 24/7 permanent cloud app is live at:")
        print(f"👉 https://huggingface.co/spaces/{repo_id}")
        print("=" * 60)
    except Exception as e:
        print(f"[ERROR] Failed uploading folder: {e}")

if __name__ == "__main__":
    main()
