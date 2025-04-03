import textacy
import spacy
from collections import defaultdict
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import unicodedata
from textacy import extract
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    # Remove extra whitespace and indentation
    text = re.sub(r'\s+', ' ', text)
    # Convert special characters to their normal equivalents
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    # Remove any remaining special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

def fetch_text_with_retries(url, max_retries=3):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
            if attempt == max_retries - 1:
                raise
    return None

def get_context_words(doc, keyword, window_size=5):
    """Get words appearing around a keyword within a specified window."""
    context_words = defaultdict(int)
    keyword = keyword.lower()
    
    for token in doc:
        if token.text.lower() == keyword:
            start = max(0, token.i - window_size)
            end = min(len(doc), token.i + window_size + 1)
            
            for i in range(start, end):
                if i != token.i and not doc[i].is_stop and not doc[i].is_punct:
                    context_words[doc[i].text.lower()] += 1
    
    return dict(context_words)

def calculate_theme_sentiment(doc, theme_keywords):
    """Calculate sentiment scores for sections containing theme keywords."""
    sentiments = []
    window_size = 10  # words before and after keyword
    
    for token in doc:
        if token.text.lower() in theme_keywords:
            start = max(0, token.i - window_size)
            end = min(len(doc), token.i + window_size + 1)
            
            # Get the sentiment of the surrounding text
            span = doc[start:end]
            sentiment = textacy.extract.basics.words(span, filter_stops=True, filter_punct=True)
            sentiments.extend([token.sentiment for token in sentiment])
    
    return np.mean(sentiments) if sentiments else 0.0

def find_similar_terms(nlp, keywords, doc, threshold=0.7):
    """Find terms semantically similar to our keywords using spaCy's word vectors."""
    similar_terms = defaultdict(set)
    
    # Get unique tokens from the document that have vectors
    doc_tokens = [token for token in doc if token.has_vector and not token.is_stop and not token.is_punct]
    
    for keyword in keywords:
        keyword_token = nlp(keyword)[0]
        if keyword_token.has_vector:
            # Use spaCy's built-in similarity
            for token in doc_tokens:
                similarity = keyword_token.similarity(token)
                if similarity > threshold:
                    similar_terms[keyword].add(token.text)
    
    return dict(similar_terms)

def analyze_text_influence():
    # Load the English language model with word vectors
    logger.info("Loading spaCy model...")
    nlp = spacy.load('en_core_web_sm')
    
    # Define the authors and their works
    authors = {
        'Mary Shelley': {
            'work': 'Frankenstein',
            'year': 1818,
            'url': 'https://www.gutenberg.org/files/84/84-0.txt'
        },
        'Lord Byron': {
            'work': 'Manfred',
            'year': 1817,
            'url': 'https://www.gutenberg.org/files/8772/8772-0.txt'
        },
        'John Polidori': {
            'work': 'The Vampyre',
            'year': 1819,
            'url': 'https://www.gutenberg.org/files/6087/6087-h/6087-h.htm'
        }
    }
    
    # Enhanced themes and motifs
    phantasmagoriana_themes = {
        'supernatural': {
            'primary': ['ghost', 'spirit', 'phantom', 'apparition', 'supernatural', 'mysterious'],
            'secondary': ['spectral', 'ethereal', 'otherworldly', 'haunting', 'paranormal']
        },
        'gothic': {
            'primary': ['castle', 'ruin', 'dark', 'gloom', 'horror', 'terror'],
            'secondary': ['dungeon', 'crypt', 'shadow', 'decay', 'forbidding', 'dread']
        },
        'romantic': {
            'primary': ['passion', 'emotion', 'love', 'desire', 'beauty', 'sublime'],
            'secondary': ['yearning', 'longing', 'ardor', 'fervor', 'rapture']
        },
        'psychological': {
            'primary': ['madness', 'insanity', 'delusion', 'dream', 'vision', 'hallucination'],
            'secondary': ['mania', 'delirium', 'nightmare', 'fever', 'trance']
        },
        'moral': {
            'primary': ['sin', 'virtue', 'redemption', 'guilt', 'punishment', 'justice'],
            'secondary': ['transgression', 'atonement', 'conscience', 'retribution']
        }
    }
    
    # Create a document collection for analysis
    docs = {}
    
    # Fetch and process each author's work
    logger.info("Fetching and processing texts...")
    for author, info in authors.items():
        try:
            text = fetch_text_with_retries(info['url'])
            if text:
                cleaned_text = clean_text(text)
                doc = nlp(cleaned_text)
                docs[author] = doc
                logger.info(f"Successfully processed {author}'s {info['work']}")
            else:
                logger.error(f"Failed to fetch text for {author}'s {info['work']}")
        except Exception as e:
            logger.error(f"Error processing {author}'s work: {str(e)}")
    
    # Analyze each theme's influence
    results = defaultdict(dict)
    
    logger.info("Analyzing themes...")
    for theme, keywords in phantasmagoriana_themes.items():
        all_keywords = keywords['primary'] + keywords['secondary']
        for author, doc in docs.items():
            # Basic keyword counts
            keyword_counts = defaultdict(int)
            for keyword in all_keywords:
                count = len([token for token in doc if token.text.lower() == keyword.lower()])
                keyword_counts[keyword] = count
            
            # Find semantically similar terms
            similar_terms = find_similar_terms(nlp, keywords['primary'], doc)
            
            # Get context words for primary keywords
            context_analysis = {}
            for keyword in keywords['primary']:
                context_analysis[keyword] = get_context_words(doc, keyword)
            
            # Calculate theme sentiment
            sentiment_score = calculate_theme_sentiment(doc, all_keywords)
            
            # Calculate relative frequencies (per 1000 words)
            doc_length = len([token for token in doc if not token.is_punct and not token.is_space])
            relative_freq = {k: (v * 1000 / doc_length) for k, v in keyword_counts.items()}
            
            # Calculate theme relevance score with more factors
            total_occurrences = sum(keyword_counts.values())
            unique_keywords_found = len([k for k, v in keyword_counts.items() if v > 0])
            context_richness = sum(len(contexts) for contexts in context_analysis.values())
            
            # Combined relevance score
            relevance_score = (
                (total_occurrences * unique_keywords_found) / len(all_keywords) +
                (context_richness * 0.5) +
                (sentiment_score * 10)  # Weight sentiment less heavily
            )
            
            # Store results
            results[theme][author] = {
                'total_occurrences': total_occurrences,
                'unique_keywords': unique_keywords_found,
                'keyword_counts': dict(keyword_counts),
                'relative_frequencies': relative_freq,
                'similar_terms': similar_terms,
                'context_analysis': context_analysis,
                'sentiment_score': sentiment_score,
                'relevance_score': relevance_score
            }
    
    # Write analysis results
    logger.info("Writing analysis results...")
    with open('phantasmagoriana_influence.md', 'w', encoding='utf-8') as f:
        f.write("# Phantasmagoriana's Influence Analysis\n\n")
        
        # Overall influence summary
        f.write("## Overall Influence Summary\n\n")
        f.write("| Author | Total Theme Occurrences | Average Relevance Score | Average Sentiment |\n")
        f.write("|--------|------------------------|------------------------|------------------|\n")
        
        author_totals = defaultdict(lambda: {'occurrences': 0, 'score': 0, 'sentiment': []})
        for theme, author_results in results.items():
            for author, data in author_results.items():
                author_totals[author]['occurrences'] += data['total_occurrences']
                author_totals[author]['score'] += data['relevance_score']
                author_totals[author]['sentiment'].append(data['sentiment_score'])
        
        for author, totals in author_totals.items():
            avg_score = totals['score'] / len(phantasmagoriana_themes)
            avg_sentiment = np.mean(totals['sentiment'])
            f.write(f"| {author} | {totals['occurrences']} | {avg_score:.2f} | {avg_sentiment:.2f} |\n")
        
        # Detailed theme analysis
        f.write("\n## Detailed Theme Analysis\n\n")
        for theme, author_results in results.items():
            f.write(f"### {theme.title()} Theme\n\n")
            
            # Theme overview
            f.write("#### Theme Overview\n\n")
            f.write("| Author | Occurrences | Relative Frequency | Unique Keywords | Sentiment | Relevance Score |\n")
            f.write("|--------|-------------|-------------------|-----------------|-----------|----------------|\n")
            
            for author, data in author_results.items():
                avg_freq = np.mean(list(data['relative_frequencies'].values()))
                f.write(f"| {author} | {data['total_occurrences']} | {avg_freq:.2f} | {data['unique_keywords']} | {data['sentiment_score']:.2f} | {data['relevance_score']:.2f} |\n")
            
            # Keyword Distribution
            f.write("\n#### Keyword Distribution and Context\n\n")
            for author, data in author_results.items():
                f.write(f"**{author}**\n\n")
                f.write("| Keyword | Count | Relative Frequency | Top Context Words |\n")
                f.write("|---------|-------|-------------------|------------------|\n")
                
                for keyword, count in sorted(data['keyword_counts'].items(), key=lambda x: x[1], reverse=True):
                    rel_freq = data['relative_frequencies'][keyword]
                    context = data['context_analysis'].get(keyword, {})
                    top_context = ', '.join(sorted(context, key=context.get, reverse=True)[:5]) if context else 'N/A'
                    f.write(f"| {keyword} | {count} | {rel_freq:.2f} | {top_context} |\n")
                
                # Similar terms found
                if data['similar_terms']:
                    f.write("\nSemantically Similar Terms:\n")
                    for keyword, terms in data['similar_terms'].items():
                        if terms:
                            f.write(f"- {keyword}: {', '.join(terms)}\n")
                
                f.write("\n---\n\n")
    
    logger.info("Analysis complete! Results saved to phantasmagoriana_influence.md")

if __name__ == "__main__":
    analyze_text_influence() 