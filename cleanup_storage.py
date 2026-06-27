"""
Delete OLD ANONYMOUS stem-splitter runs from the private Hugging Face dataset.

  - Anonymous runs live under  runs/YYYY-MM-DD/<id>/  and are deleted once older
    than RETENTION_DAYS.
  - Registered users' data lives under  users/<user_id>/...  and is PERMANENT —
    this script never touches it (it only ever deletes paths under runs/).

Run on a schedule by .github/workflows/cleanup.yml.

Env:
    HF_TOKEN          token with WRITE access to the dataset (GitHub Actions secret).
    STORAGE_DATASET   dataset id (default vincewin/stem-worker-data).
    RETENTION_DAYS    keep anonymous runs newer than this many days (default 14).
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

    # Registered-user data lives under users/ and is PERMANENT — never deleted here.
    users = {f.split("/")[1] for f in files if f.startswith("users/") and "/" in f[len("users/"):]}
    print(f"Preserving data for {len(users)} registered user(s) (users/ is never deleted).")

    # Only anonymous runs (runs/YYYY-MM-DD/...) are eligible for deletion.
    old_dates = set()
    for f in files:
        parts = f.split("/")
        if parts[0] != "runs" or len(parts) < 2:
            continue
        try:
            d = datetime.date.fromisoformat(parts[1])
        except ValueError:
            continue
        if d < cutoff:
            old_dates.add(parts[1])

    if not old_dates:
        print(f"No anonymous runs older than {DAYS} days (cutoff {cutoff}). Up to date.")
        return

    for d in sorted(old_dates):
        target = f"runs/{d}"
        # Safety net: never delete anything outside the anonymous runs/ tree.
        if not target.startswith("runs/"):
            raise RuntimeError(f"refusing to delete non-anonymous path: {target}")
        api.delete_folder(path_in_repo=target, repo_id=DATASET, repo_type="dataset",
                          commit_message=f"cleanup: remove anonymous runs from {d} (older than {DAYS} days)")
        print(f"deleted {target}")


if __name__ == "__main__":
    main()
