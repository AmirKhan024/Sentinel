"""Test fixtures that are literal Python rather than data files.

The repository's .gitignore excludes *.csv and *.jsonl project-wide, so a
data-file fixture would silently fail to commit. Keeping real discovered cases
as Python also lets each one carry the prose explaining what it protects.
"""
