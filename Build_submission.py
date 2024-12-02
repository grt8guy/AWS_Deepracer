"""
Build submission for evaluation script

Place your:
- 5-trail evaluation log tar.gz
- model tar.gz
- report.ipynb

into submission/, then fill in your name at john_w_braunsdorf = , and then run this script
>>> python build_submission.py
"""

import io
from pathlib import Path
import re
import tarfile

# Add your first and last name, lowercase, delimited with a single "-"
# e.g., firstname-lastname
YOUR_FULL_NAME = "john-braunsdorf"

# Determine if name is in the proper format
if not bool(re.compile(r'^[a-z]+-[a-z]+$').match(YOUR_FULL_NAME.lower())):
    print(f"ERROR: Your name '{YOUR_FULL_NAME}' is not of format, *lowercase* <firstname-lastname>. Please fix.")
    quit()

# Get absolute path to submission_dir
submission_dir = Path("__file__").parent.resolve() / "submission"

# Create a BytesIO object to hold the combined tar.gz in memory
output_io = io.BytesIO()

# Create a new tarfile for writing to the in-memory BytesIO object
with tarfile.open(fileobj=output_io, mode="w:gz") as output_tar:
    print("Building submission with: ")

    # Walk through submission_dir directory
    for entry in submission_dir.iterdir():

        # Work through the evaluation and model tar's
        if "eval" in entry.name or "model" in entry.name:
            with tarfile.open(entry, "r:gz") as tar:
                for member in tar.getmembers():
                    if Path(member.name).suffix in [".json", ".csv", ".pb"]:
                        info = tarfile.TarInfo(name=member.name)
                        info.size = member.size
                        output_tar.addfile(tarinfo=info, fileobj=tar.extractfile(member))
                        print(f"    -- {member.name}")

    # If you use another format, such as word or latex, replace the ipynb here with your file.
    output_tar.add(submission_dir / 'report.ipynb', arcname=submission_dir / 'report.ipynb')

# Seek to the beginning of the BytesIO object to read its content
output_io.seek(0)

# Write the new submission_dir tar
submission_tar = submission_dir / f"{YOUR_FULL_NAME}-submission.tar.gz"
with open(submission_tar, "wb") as f_out:
    f_out.write(output_io.getvalue())

print(f"Built: {submission_tar}")
