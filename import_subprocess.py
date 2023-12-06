import subprocess

# Read the list of changed files from the text file
with open('changed_files.txt', 'r') as file:
    changed_files = file.read().splitlines()

# Define the maximum number of files per commit
files_per_commit = 10  # Adjust this value based on your requirements

# Split the list of changed files into smaller chunks
file_chunks = [changed_files[i:i + files_per_commit] for i in range(0, len(changed_files), files_per_commit)]

# Iterate through the file chunks and create smaller commits
for idx, chunk in enumerate(file_chunks):
    # Stage the files for the commit
    subprocess.run(['git', 'add'] + chunk)

    # Commit the changes
    commit_message = f"Split commit {idx + 1}"
    subprocess.run(['git', 'commit', '-m', commit_message])

# Push the changes to the remote repository
subprocess.run(['git', 'push', '--force', 'https://HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/counito/EcoSentry', 'main'])
