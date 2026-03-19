from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="~/.cache/huggingface/lerobot/Qu3tzal/libero-pi0",
    repo_id="scylearning/lerobot_libero",
    repo_type="dataset",
)
