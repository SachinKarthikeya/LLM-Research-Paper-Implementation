from transformers import pipeline

# Load pre-trained summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Your long text input
text = """
Language models are a key component of many modern NLP systems. They are trained to predict the next word in a sequence given the previous words, enabling applications such as text generation, machine translation, and question answering. 
Recent advancements in transformer-based models like BERT and GPT have significantly pushed the boundaries of what language models can achieve, thanks to their ability to model long-range dependencies and large-scale training on massive datasets.
"""

# Get the summary
summary = summarizer(text, max_length=60, min_length=25, do_sample=False)

# Print the summary
print("Summary:", summary[0]['summary_text'])