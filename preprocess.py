
import re
# lightweight stopwords list to avoid external downloads
STOPWORDS = set([
    'a','an','the','is','it','this','that','and','or','of','to','in','for','on','with','as','are','was','were','be','by','i','you','he','she','they','we','my','your'
])
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\\S+|www\\S+', '', text)
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t)>1]
    return ' '.join(tokens)
