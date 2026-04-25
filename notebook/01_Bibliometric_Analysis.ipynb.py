# ==============================================================================
# 01_Bibliometric_Analysis
# Compiled Script for PM2.5 & Machine Learning Bibliometric Analysis
# ==============================================================================

# --- CELL 1: Import Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import networkx as nx
from collections import Counter
import itertools
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")


# --- CELL 2: Data Loading and Preprocessing ---
print("Initializing Data Pipeline...")
try:
    # Attempt to load actual Web of Science export file
    df = pd.read_csv('wos_raw_data.csv', encoding='utf-8-sig')
    print("Real data loaded successfully.")
except FileNotFoundError:
    print("wos_raw_data.csv not found. Generating mock data for demonstration...")
    # Mock data to simulate Web of Science extract
    mock_data = {
        'Publication Year': [2018, 2019, 2019, 2020, 2020, 2020, 2021, 2021, 2021, 2021, 2022, 2022, 2022, 2022, 2022, 2023, 2023, 2023],
        'Journal Name': ['Atmosphere', 'Science of The Total Environment', 'Atmospheric Environment', 'Remote Sensing', 'Environmental Pollution', 'Atmosphere', 'Science of The Total Environment', 'Urban Climate', 'Sustainability', 'Atmospheric Environment', 'IEEE Access', 'Scientific Reports', 'Atmosphere', 'Science of The Total Environment', 'Environmental Pollution', 'Atmospheric Pollution Research', 'Environment International', 'Atmosphere'],
        'Author Keywords': ['Machine Learning; PM2.5; Air Quality', 'Deep Learning; LSTM; Time Series', 'Random Forest; PM2.5; Meteorology', 'Machine Learning; Remote Sensing; AOD', 'Air Pollution; Neural Network; PM2.5', 'XGBoost; Forecasting; Urban Climate', 'Deep Learning; PM2.5; Air Quality', 'Machine Learning; COVID-19; Air Pollution', 'LSTM; Spatiotemporal; PM2.5', 'Machine Learning; Feature Importance; PM2.5'] * 2 + ['Random Forest; Air Quality; Prediction'] * -2,
        'Document Title': [f'Paper {i}' for i in range(18)]
    }
    df = pd.DataFrame(mock_data)

# Preprocessing: Remove duplicates based on Document Title
initial_len = len(df)
df = df.drop_duplicates(subset=['Document Title'])
print(f"Removed {initial_len - len(df)} duplicate records.")

# Ensure consistency in keywords
df['Author Keywords'] = df['Author Keywords'].fillna('').str.upper()


# --- CELL 3: Temporal Performance Analysis (Publication Growth) ---
print("\nGenerating Publication Growth Chart...")
plt.figure(figsize=(10, 5))
yearly_counts = df['Publication Year'].value_counts().sort_index()

plt.fill_between(yearly_counts.index, yearly_counts.values, color="skyblue", alpha=0.4)
plt.plot(yearly_counts.index, yearly_counts.values, color="Slateblue", marker="o", linewidth=2, markersize=6)

plt.title('Publication Growth Over Years (PM2.5 & Machine Learning)', fontsize=14, fontweight='bold')
plt.xlabel('Publication Year', fontsize=12)
plt.ylabel('Number of Publications', fontsize=12)
plt.xticks(yearly_counts.index, rotation=45)
plt.tight_layout()
plt.show()


# --- CELL 4: Source Impact Evaluation (Top 10 Journals) ---
print("\nGenerating Top Journals Chart...")
plt.figure(figsize=(10, 6))
top_journals = df['Journal Name'].str.upper().value_counts().head(10)

sns.barplot(x=top_journals.values, y=top_journals.index, palette='viridis')

plt.title('Top Most Prolific Scientific Journals', fontsize=14, fontweight='bold')
plt.xlabel('Number of Articles', fontsize=12)
plt.ylabel('Journal Name', fontsize=12)

for i, v in enumerate(top_journals.values):
    plt.text(v + 0.1, i, str(v), color='black', va='center')

plt.tight_layout()
plt.show()


# --- CELL 5: Subject Keyword Mining (Word Cloud) ---
print("\nGenerating Keyword Word Cloud...")
all_keywords = df['Author Keywords'].str.split(';').explode().str.strip()
all_keywords = all_keywords[all_keywords != '']
keyword_freq = Counter(all_keywords)

wordcloud = WordCloud(width=800, height=400, 
                      background_color='white', 
                      colormap='Dark2',
                      max_words=100).generate_from_frequencies(keyword_freq)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Author Keywords Word Cloud', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# --- CELL 6: Topology Network Structure (Co-occurrence) ---
print("\nGenerating Topology Network Structure...")
plt.figure(figsize=(12, 8))

edges = []
for keywords in df['Author Keywords'].str.split(';'):
    cleaned_keywords = [k.strip() for k in keywords if k.strip()]
    if len(cleaned_keywords) > 1:
        edges.extend(itertools.combinations(cleaned_keywords, 2))

G = nx.Graph()
for u, v in edges:
    if G.has_edge(u, v):
        G[u][v]['weight'] += 1
    else:
        G.add_edge(u, v, weight=1)

core_nodes = [node for node, degree in dict(G.degree()).items() if degree >= 1]
G_core = G.subgraph(core_nodes)

pos = nx.spring_layout(G_core, k=0.5, seed=42)

node_sizes = [dict(G_core.degree())[n] * 300 for n in G_core.nodes()]
nx.draw_networkx_nodes(G_core, pos, node_size=node_sizes, node_color='lightskyblue', alpha=0.8, edgecolors='black')

edge_weights = [G_core[u][v]['weight'] * 1 for u, v in G_core.edges()]
nx.draw_networkx_edges(G_core, pos, width=edge_weights, edge_color='gray', alpha=0.5)

nx.draw_networkx_labels(G_core, pos, font_size=8, font_weight='bold')

plt.title('Topology Network Structure: Co-occurrence of Research Keywords', fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

density = nx.density(G_core)
clustering_coefficient = nx.average_clustering(G_core)
print(f"Network Density: {density:.4f}")
print(f"Average Clustering Coefficient: {clustering_coefficient:.4f}")
print("\nExecution Complete.")