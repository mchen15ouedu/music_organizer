"""
Delete archived stem-splitter runs older than RETENTION_DAYS from the private
Hugging Face dataset. Run on a schedule by .github/workflows/cleanup.yml.

Runs are stored under runs/YYYY-MM-DD/<id>/..., so cleanup just parses the date
in the path and deletes whole date folders past the cutoff.

Env:
    HF_TOKEN          token with WRITE access to the dataset (GitHub Actions secret).
    STORAGE_DATASET   dataset id (default vincewin/stem-worker-data).
    RETENTION_DAYS    keep runs newer than this many days (default 14).
"""
import os
import datetime
from huggingface_hub import HfApi

DATASET = os.environ.get("STORAGE_DATASET", "vincewin/stem-worker-data")
DAYS = int(os.environ.get("RETENTION_DAYS", "14"))


def main():
    api = HfApi(token=os.environ["HF_TOKEN"])
    files = api.list_repo_files(repo_id=DATASET, repo_type="dataset")
    cutoff = datetime.date.today() - datetime.timedelta(days=DAYS)
    old_dates = set()
    for f in files:
        parts = f.split("/")
        if len(parts) >= 2 and parts[0] == "runs":
            try:
                d = datetime.date.fromisoformat(parts[1])
            except ValueError:
                continue
            if d < cutoff:
                old_dates.add(parts[1])

    if not old_dates:
        print(f"Nothing older than {DAYS} days (cutoff {cutoff}). Up to date.")
        return
    for d in sorted(old_dates):
        api.delete_folder(path_in_repo=f"runs/{d}", repo_id=DATASET, repo_type="dataset",
                          commit_message=f"cleanup: remove runs from {d} (older than {DAYS} days)")
        print(f"deleted runs/{d}")


if __name__ == "__main__":
    main()
