import re

def clean_text(text):
    # convert to lowercase
    text = text.lower()
    
    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)
    
    # remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text