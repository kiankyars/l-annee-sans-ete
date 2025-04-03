# Literary Analysis Project

This project consists of three main components analyzing different aspects of early 19th-century literature:

1. Phantasmagoriana's influence on Gothic literature
2. Correspondence network analysis of literary figures
3. Theme analysis across works
4. Geographic visualization of literary connections

## Project Structure

```
.
├── correspondence_analysis/    # Analysis of literary correspondence
│   ├── data/
│   │   ├── letters.html
│   │   └── Smiles: Memoir of John Murray.html
│   ├── results/
│   │   ├── correspondence_network.png
│   │   └── filtered_letters.html
│   └── src/
│       └── hello.py
│
├── map_visualization/         # Geographic visualization
│   ├── data/
│   ├── results/
│   │   └── map.html
│   └── src/
│
├── phantasmagoriana_analysis/ # Literary influence analysis
│   ├── data/
│   ├── results/
│   │   ├── phantasmagoriana_influence.md
│   │   └── moral_keywords.txt
│   └── src/
│       └── phantasmagoriana_analysis.py
│
├── theme_analysis/           # Thematic analysis
│   ├── data/
│   ├── results/
│   │   └── theme_analysis.txt
│   └── src/
│
├── README.md
├── pyproject.toml
└── requirements.txt

```

## Components

### 1. Phantasmagoriana Analysis
- Analyzes the influence of Phantasmagoriana on Gothic literature
- Focuses on works by Shelley, Byron, and Polidori
- Includes thematic and semantic analysis

### 2. Correspondence Analysis
- Network analysis of literary correspondence
- Visualizes connections between authors
- Processes historical letters and memoirs

### 3. Theme Analysis
- Identifies and analyzes recurring themes
- Keyword analysis and context examination
- Cross-work thematic comparison

### 4. Map Visualization
- Geographic visualization of literary connections
- Maps locations mentioned in correspondence
- Spatial analysis of literary networks

## Usage

Each component has its own source code and can be run independently. See individual component directories for specific instructions.

### Dependencies
Install all required packages:
```bash
uv add spacy textacy beautifulsoup4 requests tqdm numpy networkx matplotlib
uv run python -m spacy download en_core_web_sm
```

## Data Sources
- Project Gutenberg texts
- Historical correspondence
- John Murray's memoir
- Geographic data

## Analysis Features
- Network analysis
- Thematic analysis
- Geographic visualization
- Sentiment analysis
- Semantic similarity detection
