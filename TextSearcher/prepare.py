CNN_FRACTION = 0.05
DAILYMAIL_FRACTION = 0.05

import csv
import hashlib
import os
import tensorflow as tf

dm_single_close_quote = u"\u2019"  # unicode
dm_double_close_quote = u"\u201d"
END_TOKENS = [
    ".", "!", "?", "...", "'", "`", '"', dm_single_close_quote,
    dm_double_close_quote, ")"
]  # acceptable ways to end a sentence


def read_file(file_path):
  """Reads lines in the file."""
  lines = []
  with tf.io.gfile.GFile(file_path, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def url_hash(url):
  """Gets the hash value of the url."""
  h = hashlib.sha1()
  url = url.encode("utf-8")
  h.update(url)
  return h.hexdigest()


def get_url_hashes_dict(urls_path):
  """Gets hashes dict that maps the hash value to the original url in file."""
  urls = read_file(urls_path)
  return {url_hash(url): url[url.find("id_/") + 4:] for url in urls}


def find_files(folder, url_dict):
  """Finds files corresponding to the urls in the folder."""
  all_files = tf.io.gfile.listdir(folder)
  ret_files = []
  for file in all_files:
    # Gets the file name without extension.
    filename = os.path.splitext(os.path.basename(file))[0]
    if filename in url_dict:
      ret_files.append(os.path.join(folder, file))
  return ret_files


def fix_missing_period(line):
  """Adds a period to a line that is missing a period."""
  if "@highlight" in line:
    return line
  if not line:
    return line
  if line[-1] in END_TOKENS:
    return line
  return line + "."


def get_highlights(story_file):
  """Gets highlights from a story file path."""
  lines = read_file(story_file)

  # Put periods on the ends of lines that are missing them
  # (this is a problem in the dataset because many image captions don't end in
  # periods; consequently they end up in the body of the article as run-on
  # sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  highlight_list = []
  next_is_highlight = False
  for line in lines:
    if not line:
      continue  # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlight_list.append(line)

  # Make highlights into a single string.
  highlights = "\n".join(highlight_list)

  return highlights

url_hashes_dict = get_url_hashes_dict("all_train.txt")
cnn_files = find_files("cnn/stories", url_hashes_dict)
dailymail_files = find_files("dailymail/stories", url_hashes_dict)

# The size to be selected.
cnn_size = int(CNN_FRACTION * len(cnn_files))
dailymail_size = int(DAILYMAIL_FRACTION * len(dailymail_files))
print("CNN size: %d"%cnn_size)
print("Daily Mail size: %d"%dailymail_size)

with open("cnn_dailymail.csv", "w") as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=["highlights", "urls"])
  writer.writeheader()

  for file in cnn_files[:cnn_size] + dailymail_files[:dailymail_size]:
    highlights = get_highlights(file)
    # Gets the filename which is the hash value of the url.
    filename = os.path.splitext(os.path.basename(file))[0]
    url = url_hashes_dict[filename]
    writer.writerow({"highlights": highlights, "urls": url})