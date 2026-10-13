
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt

df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[["v1", "v2"]]
df.columns = ["label", "message"]

df["label"] = df["label"].str.lower()

spam_messages = df[df["label"] == "spam"]["message"]

stop_words = {
    "the", "a", "an", "is", "to", "and", "in", "you",
    "for", "of", "on", "your", "it", "this", "that",
    "are", "be", "with", "i", "me", "my", "we", "at",
    "have", "has", "will", "can", "or", "our", "so",
    "if", "but", "do", "not", "from", "up", "now"
}

word_counts = Counter()

for message in spam_messages:
    words = re.findall(r"\b[a-zA-Z]+\b", str(message).lower())
    words = [word for word in words if word not in stop_words and len(word) > 2]
    word_counts.update(words)

top_words = word_counts.most_common(15)

words = [item[0] for item in top_words]
counts = [item[1] for item in top_words]

plt.figure(figsize=(10, 6))
plt.barh(words[::-1], counts[::-1], color="#4285F4")
plt.xlabel("Number of occurrences")
plt.ylabel("Words")
plt.title("Top 15 Common Words in Spam Messages")
plt.tight_layout()
plt.savefig("spam_keywords.png", dpi=150)
plt.show()