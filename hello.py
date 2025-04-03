from bs4 import BeautifulSoup
import re
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from collections import defaultdict
import requests

def extract_name_from_text(text):
    # Extract the name from the beginning of the letter text
    # Format is typically "Name to Recipient, date:"
    match = re.match(r'([^,]+) to ([^,]+),', text)
    if match:
        return match.group(1).strip()
    return text.split(' to ')[0].strip()

def is_relevant_person(name):
    relevant_people = [
        'LdByron',    # Lord Byron
        'JoPolid1821'  # John Polidori
    ]
    return name in relevant_people

def fetch_letter_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Get all paragraph elements
            paragraphs = soup.find_all('p')
            
            # Extract text from each paragraph
            letter_content = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text:
                    # Basic cleanup
                    text = re.sub(r'A\. D\. \d{4}\..*?\d+', '', text)  # Remove page numbers
                    text = re.sub(r'LIFE OF LORD BYRON\.', '', text)  # Remove headers
                    text = text.strip()
                    if text:
                        letter_content.append(text)
            
            # Join all paragraphs with proper spacing
            if letter_content:
                return '\n\n'.join(letter_content)
                
    except Exception as e:
        print(f"Warning: Could not fetch content from {url}: {str(e)}")
    return None

def parse_letters(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    letters = []
    name_mapping = {}  # Maps IDs to actual names
    
    for div in soup.find_all('div', attrs={'style': 'margin-left: 30px; text-indent: -30px;'}):
        writer = div.get('writer', '')
        recipient = div.get('recipient', '')
        text = div.get_text(strip=True)
        
        # Extract source URL from the text
        source_url = None
        if 'href=' in str(div):
            link = div.find('a')
            if link and 'href' in link.attrs:
                source_url = link['href']
                if not source_url.startswith('http'):
                    source_url = f"https://lordbyron.org/{source_url}"
        
        # Extract actual names
        writer_name = extract_name_from_text(text)
        
        # More robust recipient name extraction
        try:
            if ' to ' in text:
                recipient_name = text.split(' to ')[1].split(',')[0].strip()
            else:
                recipient_name = recipient
        except IndexError:
            print(f"Warning: Could not parse recipient from text: {text[:100]}...")
            recipient_name = recipient
        
        # Store name mappings
        name_mapping[writer] = writer_name
        name_mapping[recipient] = recipient_name
        
        # Check if either the writer or recipient is one of our relevant people
        if is_relevant_person(writer) or is_relevant_person(recipient):
            date = div.get('date', '')
            
            # Fetch the actual letter content
            letter_content = None
            if source_url:
                print(f"Fetching content from: {source_url}")
                letter_content = fetch_letter_content(source_url)
            
            letters.append({
                'writer': writer,
            'recipient': recipient,
                'writer_name': writer_name,
                'recipient_name': recipient_name,
                'date': date,
                'text': text,
                'content': letter_content or text,  # Use fetched content if available, fallback to header text
                'source_url': source_url
            })
            print(f"Found relevant letter: {text[:100]}...")

    return letters, name_mapping

def create_correspondence_network(letters, name_mapping):
    G = nx.DiGraph()
    
    # Count letters between each pair and track dates
    edge_counts = defaultdict(int)
    edge_dates = defaultdict(list)
    
    for letter in letters:
        writer = name_mapping[letter['writer']]
        recipient = name_mapping[letter['recipient']]
        date_str = letter['date']
        
        # Create separate nodes for sender and recipient
        writer_node = f"{writer}_sender"
        recipient_node = f"{recipient}_recipient"
        
        # Count letters between pairs
        edge_counts[(writer_node, recipient_node)] += 1
        
        # Store dates if available
        if date_str:
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                edge_dates[(writer_node, recipient_node)].append(date)
            except ValueError:
                print(f"Warning: Could not parse date: {date_str}")
    
    # Add edges with weights and calculate date ranges
    for (writer_node, recipient_node), count in edge_counts.items():
        dates = edge_dates[(writer_node, recipient_node)]
        if dates:
            date_range = f"{min(dates).strftime('%b %d')} - {max(dates).strftime('%b %d')}"
        else:
            date_range = "No dates"
            
        G.add_edge(writer_node, recipient_node, 
                  weight=count,
                  date_range=date_range,
                  dates=dates)
    
    # Create the visualization with a larger figure size
    plt.figure(figsize=(20, 15))
    
    # Use spring layout with increased k parameter for better spacing
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes with different colors for senders and recipients
    sender_nodes = [node for node in G.nodes() if node.endswith('_sender')]
    recipient_nodes = [node for node in G.nodes() if node.endswith('_recipient')]
    
    # Draw sender nodes with larger size
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=sender_nodes,
                          node_color='lightblue',
                          node_size=4000)
    
    # Draw recipient nodes with larger size
    nx.draw_networkx_nodes(G, pos,
                          nodelist=recipient_nodes,
                          node_color='lightgreen',
                          node_size=4000)
    
    # Draw edges with varying widths based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights)
    normalized_weights = [4 * w/max_weight for w in weights]  # Increased edge width
    
    # Draw edges with arrows and increased spacing
    nx.draw_networkx_edges(G, pos,
                          width=normalized_weights,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=40,  # Larger arrows
                          arrowstyle='->',
                          connectionstyle='arc3,rad=0.3')  # Increased curve
    
    # Add node labels with larger font
    labels = {node: node.replace('_sender', '').replace('_recipient', '') 
             for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
    
    # Add edge labels with improved formatting
    edge_labels = {(u, v): f"{G[u][v]['weight']} letters\n{G[u][v]['date_range']}"
                  for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    
    plt.title("Letter Correspondence Network (1816)\nEdge width represents number of letters, labels show count and date range\nBlue nodes: Senders, Green nodes: Recipients", 
              fontsize=14, pad=20)
    plt.savefig('correspondence_network.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_themes(letters):
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Define theme keywords
    themes = {
        'summer': ['summer', 'weather', 'climate', 'cold', 'rain', 'storm', 'sun', 
                  'smoke', 'clouds', 'volcano', 'eruption', 'darkness', 'fog', 
                  'temperature', 'atmosphere', 'gloom', 'dismal', 'unusual weather'],
        'romantic': ['romantic', 'poetry', 'nature', 'beauty', 'sublime', 'passion',
                    'emotion', 'imagination', 'inspiration', 'art', 'literature',
                    'creative', 'expression', 'feelings', 'love', 'desire'],
        'gothic': ['ghost', 'monster', 'horror', 'dark', 'supernatural', 'frankenstein',
                  'mystery', 'terror', 'haunted', 'death', 'morbid', 'melancholy',
                  'gloom', 'despair', 'night', 'shadow', 'creature', 'fear']
    }
    
    # Extract all letter contents
    texts = [letter['content'] for letter in letters]
    
    # Get embeddings for all texts
    embeddings = model.encode(texts)
    
    # Analyze each theme
    theme_scores = {}
    for theme, keywords in themes.items():
        # Create a query embedding for the theme
        query = ' '.join(keywords)
        query_embedding = model.encode([query])[0]
        
        # Calculate similarity scores
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Store scores
        theme_scores[theme] = similarities
    
    # Write analysis results to file
    with open('theme_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("Theme Analysis Results\n")
        f.write("=====================\n\n")
        
        for theme, scores in theme_scores.items():
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            f.write(f"{theme.capitalize()} Theme:\n")
            f.write(f"Average relevance: {avg_score:.3f}\n")
            f.write(f"Most relevant letter (score: {max_score:.3f}):\n")
            max_idx = np.argmax(scores)
            f.write(f"- From: {letters[max_idx]['writer_name']} to {letters[max_idx]['recipient_name']}\n")
            f.write(f"- Date: {letters[max_idx]['date']}\n")
            f.write(f"- Source: {letters[max_idx]['source_url']}\n")
            f.write("\nFull Letter Content:\n")
            f.write("-" * 80 + "\n")
            f.write(letters[max_idx]['content'])
            f.write("\n" + "-" * 80 + "\n\n")
            
            # Add top 3 most relevant letters for each theme
            top_indices = np.argsort(scores)[-3:][::-1]
            f.write("Top 3 most relevant letters:\n")
            for idx in top_indices:
                f.write(f"\n- Score: {scores[idx]:.3f}\n")
                f.write(f"  From: {letters[idx]['writer_name']} to {letters[idx]['recipient_name']}\n")
                f.write(f"  Date: {letters[idx]['date']}\n")
                f.write(f"  Source: {letters[idx]['source_url']}\n")
                f.write("\n  Full Letter Content:\n")
                f.write("  " + "-" * 76 + "\n")
                f.write("  " + letters[idx]['content'].replace('\n', '\n  '))
                f.write("\n  " + "-" * 76 + "\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    print("Theme analysis results have been saved to theme_analysis.txt")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Analyze historical letters')
    parser.add_argument('--mode', choices=['graph', 'analyze', 'both'], 
                      default='both',
                      help='Mode of operation: graph (correspondence network only), '
                           'analyze (theme analysis only), or both')
    args = parser.parse_args()
    
    # Read the input HTML file
    with open('letters.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Parse and filter the letters
    letters, name_mapping = parse_letters(html_content)
    
    # Create the correspondence network visualization if requested
    if args.mode in ['graph', 'both']:
        create_correspondence_network(letters, name_mapping)
        print("Created correspondence network visualization (correspondence_network.png)")
    
    # Save filtered letters to a new HTML file
    with open('filtered_letters.html', 'w', encoding='utf-8') as f:
        for letter in letters:
            f.write(f'<div date="{letter["date"]}" recipient="{letter["recipient"]}" writer="{letter["writer"]}" style="margin-left: 30px; text-indent: -30px;">{letter["text"]}\n</div>\n')
    
    print(f"\nFound {len(letters)} relevant letters. Saved to filtered_letters.html")
    
    # Perform theme analysis if requested
    if args.mode in ['analyze', 'both']:
        analyze_themes(letters)

if __name__ == "__main__":
    main()