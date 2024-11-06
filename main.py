import urllib.request
import sys
from unicodedata import category
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import markovify
import nltk

nltk.download('vader_lexicon')

def read_text_from_url(url):
    """Fetches and returns text from a URL."""
    with urllib.request.urlopen(url) as f:
        return f.read().decode('utf-8')

def process_text(text, skip_header):
    """Makes a histogram that counts the words from a given text.

    text: string containing the entire text to process
    skip_header: boolean, whether to skip the Gutenberg header

    returns: map from each word to the number of times it appears.
    """
    hist = {}

    if skip_header:
        text = skip_gutenberg_header(text)

    strippables = "".join(
        chr(i) for i in range(sys.maxunicode) if category(chr(i)).startswith("P")
    )

    for line in text.splitlines():
        if line.startswith("*** END OF THE PROJECT"):
            break

        line = line.replace("-", " ")
        line = line.replace(chr(8212), " ") 

        for word in line.split():
            word = word.strip(strippables)
            word = word.lower()

            hist[word] = hist.get(word, 0) + 1

    return hist

def skip_gutenberg_header(text):
    """Removes the header from the Gutenberg text, returning text from the main content onward."""
    start_marker = "START OF THE PROJECT"
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if start_marker.lower() in line.lower(): 
            return "\n".join(lines[i+1:])
    
    # If the start marker was not found
    raise ValueError(f"Header end marker '{start_marker}' not found in text.")

def most_common(hist, excluding_stopwords=False):
    """Makes a list of word-freq pairs in descending order of frequency.
    
    hist: map from word to frequency
    returns: list of (frequency, word) pairs
    """
    stopwords = {
        'the', 'and', 'to', 'of', 'a', 'i', 'it', 'in', 'or', 'is', 'd', 's', 'that',
        'you', 'he', 'she', 'we', 'they', 'not', 'this', 'but', 'on', 'with', 'for',
        'as', 'at', 'by', 'from', 'about', 'was', 'be', 'are', 'have', 'has', 'had', 'her',
        'his', 'him', 'my', 'all', 'so', 'which', 'were', ' ', '', 'their', 'there', 'if', 'an',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'one', 'two', 'three', 'four', 'five',
    }

    freq_word_tuples = []
    for word, freq in hist.items():
        if excluding_stopwords and word in stopwords:
            continue
        freq_word_tuples.append((freq, word))
    
    freq_word_tuples.sort(reverse=True)
    return freq_word_tuples


def print_most_common(hist, num=10):
    """Prints the most commons words in a histogram and their frequencies.
    
    hist: histogram (map from word to frequency)
    num: number of words to print
    """
    common_words = most_common(hist, excluding_stopwords=True)
    for freq, word in common_words[:num]:
        print(f"{word}: {freq}")


def analyze_sentiment(text):
    """Analyzes the sentiment of each sentence in the text."""
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    for sentence in text.splitlines():
        if sentence.strip():  
            score = sia.polarity_scores(sentence)
            sentiments.append((sentence, score))
    return sentiments


def calculate_similarity(text1, text2):
    """Calculates cosine similarity between two texts."""
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)
    return similarity[0][1] 

def visualize_text_clusters(texts):
    """Visualizes clusters of texts based on their cosine similarity using MDS."""
    vectorizer = CountVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    similarities = cosine_similarity(vectors)
    dissimilarities = 1 - similarities  
    mds = MDS(dissimilarity='precomputed')
    coords = mds.fit_transform(dissimilarities)
    
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, text in enumerate(texts):
        plt.annotate(f"Text {i+1}", (coords[i, 0], coords[i, 1]))
    
    plt.title("Text Clusters")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()


def generate_text_markov(text, num_sentences=5):
    """Generates new sentences based on input text using a Markov chain."""
    text_model = markovify.Text(text)
    generated_text = []
    for _ in range(num_sentences):
        sentence = text_model.make_sentence()
        if sentence:
            generated_text.append(sentence)
    return " ".join(generated_text)


def main():
    url = 'https://www.gutenberg.org/cache/epub/13177/pg13177.txt'
    text = read_text_from_url(url)
    
    hist = process_text(text, skip_header=True)
    print("Word Frequency Histogram:")
    print(hist)
    
    print("\nMost Common Words:")
    print_most_common(hist, 10)

    print("\nSentiment Analysis:")
    sentiments = analyze_sentiment(text)
    for sentence, score in sentiments[45:60]:  # Show some interesting sentences
        print(f"Sentence: {sentence}\nSentiment Score: {score}\n")

    comparison_url = "https://www.gutenberg.org/cache/epub/65061/pg65061.txt"
    comparison_text = read_text_from_url(comparison_url)

    similarity_score = calculate_similarity(text, comparison_text)  # This would compare two different texts
    print(f"\nCosine Similarity (between identical texts for demo): {similarity_score}")

    visualize_text_clusters([text, comparison_text])

    generated_text = generate_text_markov(text)
    print("\nGenerated Text with Markov Chains:")
    print(generated_text)

if __name__ == "__main__":
    main()
