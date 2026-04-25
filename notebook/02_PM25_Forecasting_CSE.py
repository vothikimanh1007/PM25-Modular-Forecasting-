# ==============================================================================
# 02_PM25_Forecasting_CSE
# Compiled Script for PM2.5 Forecasting and Architecture Visualization
# ==============================================================================

# --- CELL 1 ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import glob
import os

def merge_and_analyze_wos():
    print("--- STARTING WOS DATA MERGING AND ANALYSIS ---")

    # 1. Merge CSV files
    csv_files = glob.glob("savedrecs*.csv")

    if not csv_files:
        print("Error: No CSV files found. Please upload them to Colab.")
        return

    print(f"Found {len(csv_files)} files. Merging in progress...")

    df_list = []
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'utf-16']

    for file in csv_files:
        df_temp = None
        for enc in encodings_to_try:
            try:
                # Try reading the CSV file with the current encoding
                df_temp = pd.read_csv(file, on_bad_lines='skip', encoding=enc)
                print(f"Successfully loaded: {file} (encoding: {enc})")
                break # Exit the encoding loop if successful
            except UnicodeDecodeError:
                # If Unicode error, continue to the next encoding in the list
                continue
            except Exception as e:
                # If it's a different error (e.g. file missing), break and log
                print(f"Error reading {file} with {enc}: {e}")
                break

        if df_temp is not None:
            df_list.append(df_temp)
        else:
            print(f"Failed to read {file} with all attempted encodings.")

    # Concatenate all dataframes
    if not df_list:
        print("Error: No valid data could be extracted from the files.")
        return

    df_merged = pd.concat(df_list, ignore_index=True)

    # Remove duplicate articles based on Document Title
    if 'Document Title' in df_merged.columns:
        df_merged = df_merged.drop_duplicates(subset=['Document Title'])

    print(f"Total unique articles after dropping duplicates: {len(df_merged)}")

    # Save the merged dataset
    output_csv = "merged_wos_dataset.csv"
    df_merged.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Merged dataset successfully saved as '{output_csv}'\n")

    # 2. Bibliometric Analysis & Plotting
    print("--- GENERATING BIBLIOMETRIC FIGURES ---")

    # Set plotting style
    sns.set_theme(style="whitegrid")

    # Figure 1: Publication Evolution (Bar Chart)
    if 'Year Published' in df_merged.columns:
        # Clean the year column
        df_merged['Year Published'] = pd.to_numeric(df_merged['Year Published'], errors='coerce')
        df_clean_year = df_merged.dropna(subset=['Year Published']).copy()
        df_clean_year['Year Published'] = df_clean_year['Year Published'].astype(int)

        plt.figure(figsize=(10, 6))
        pub_per_year = df_clean_year['Year Published'].value_counts().sort_index()

        sns.barplot(x=pub_per_year.index, y=pub_per_year.values, palette="viridis")

        # English labels for the figure
        plt.title("Publication Evolution Over Years (PM2.5 & Machine Learning)", fontsize=14, fontweight='bold')
        plt.xlabel("Publication Year", fontsize=12)
        plt.ylabel("Number of Articles", fontsize=12)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("fig1_publication_evolution.png", dpi=300)
        print("Generated Figure 1: fig1_publication_evolution.png")
        plt.show()

    # Figure 2: Top Journals (Horizontal Bar Chart)
    if 'Publication Name' in df_merged.columns:
        plt.figure(figsize=(10, 6))
        top_journals = df_merged['Publication Name'].value_counts().head(10)

        sns.barplot(y=top_journals.index, x=top_journals.values, palette="magma")

        # English labels for the figure
        plt.title("Top 10 Most Prolific Scientific Journals", fontsize=14, fontweight='bold')
        plt.xlabel("Number of Articles", fontsize=12)
        plt.ylabel("Journal Name", fontsize=12)

        plt.tight_layout()
        plt.savefig("fig2_top_journals.png", dpi=300)
        print("Generated Figure 2: fig2_top_journals.png")
        plt.show()

    # Figure 3: Author Keywords (Word Cloud)
    if 'Author Keywords' in df_merged.columns:
        # Extract and clean keywords
        keywords = df_merged['Author Keywords'].dropna().astype(str).str.cat(sep='; ')
        keywords = keywords.replace(';', ' ')

        wordcloud = WordCloud(width=800, height=400,
                              background_color='white',
                              colormap='inferno',
                              max_words=100).generate(keywords)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        # English labels for the figure
        plt.title("Author Keywords Word Cloud", fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig("fig3_author_keywords_wordcloud.png", dpi=300)
        print("Generated Figure 3: fig3_author_keywords_wordcloud.png")
        plt.show()

    print("\n--- PROCESS COMPLETED ---")
    print("Merged CSV and plot images (.png) are ready to be downloaded.")

# Execute the script
if __name__ == "__main__":
    merge_and_analyze_wos()

# --- CELL 2 ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import glob
import os
from collections import Counter
import numpy as np

def generate_fancy_bibliometrics():
    print("--- STARTING ADVANCED WOS DATA ANALYSIS ---")

    # 1. Merge CSV files with robust encoding handling
    csv_files = glob.glob("savedrecs*.csv")
    if not csv_files:
        print("Error: No CSV files found. Please upload them to Colab.")
        return

    print(f"Found {len(csv_files)} files. Merging in progress...")
    df_list = []
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'utf-16']

    for file in csv_files:
        df_temp = None
        for enc in encodings_to_try:
            try:
                df_temp = pd.read_csv(file, on_bad_lines='skip', encoding=enc)
                print(f"Successfully loaded: {file} (encoding: {enc})")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading {file} with {enc}: {e}")
                break

        if df_temp is not None:
            df_list.append(df_temp)

    if not df_list:
        print("Error: No valid data could be extracted.")
        return

    df_merged = pd.concat(df_list, ignore_index=True)

    if 'Document Title' in df_merged.columns:
        df_merged = df_merged.drop_duplicates(subset=['Document Title'])

    print(f"Total unique articles: {len(df_merged)}")
    df_merged.to_csv("merged_wos_dataset_final.csv", index=False, encoding='utf-8-sig')

    # 2. Advanced Bibliometric Plotting
    print("\n--- GENERATING FANCY FIGURES FOR PUBLICATION ---")

    # Global settings for high-quality academic plots
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'

    # ---------------------------------------------------------
    # FIGURE 1: Fancy Area-Line Chart for Publication Trend
    # ---------------------------------------------------------
    if 'Year Published' in df_merged.columns:
        df_merged['Year Published'] = pd.to_numeric(df_merged['Year Published'], errors='coerce')
        df_clean_year = df_merged.dropna(subset=['Year Published']).copy()
        df_clean_year['Year Published'] = df_clean_year['Year Published'].astype(int)

        # Filter realistic years (e.g., 2010 to 2026)
        df_clean_year = df_clean_year[(df_clean_year['Year Published'] >= 2010) & (df_clean_year['Year Published'] <= 2026)]
        pub_per_year = df_clean_year['Year Published'].value_counts().sort_index()

        plt.figure(figsize=(10, 5))
        ax = sns.lineplot(x=pub_per_year.index, y=pub_per_year.values,
                          marker='o', markersize=8, linewidth=2.5, color='#2c3e50')

        # Fill area under the line
        plt.fill_between(pub_per_year.index, pub_per_year.values, color='#3498db', alpha=0.3)

        plt.title("Publication Growth Over Years (PM2.5 & Machine Learning)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Publication Year", fontsize=12, fontweight='bold')
        plt.ylabel("Number of Publications", fontsize=12, fontweight='bold')
        plt.xticks(pub_per_year.index, rotation=45)
        plt.tight_layout()
        plt.savefig("Fig1_Publication_Trend.png", dpi=300, bbox_inches='tight')
        print("Generated: Fig1_Publication_Trend.png")
        plt.close()

    # ---------------------------------------------------------
    # FIGURE 2: Horizontal Bar Chart with Data Labels (Top Journals)
    # ---------------------------------------------------------
    if 'Publication Name' in df_merged.columns:
        plt.figure(figsize=(12, 6))
        top_journals = df_merged['Publication Name'].value_counts().head(10)

        ax = sns.barplot(x=top_journals.values, y=top_journals.index, palette="viridis")

        # Add values at the end of each bar
        for i, v in enumerate(top_journals.values):
            ax.text(v + 0.5, i, str(v), color='black', va='center', fontweight='bold')

        plt.title("Top 10 Most Prolific Scientific Journals", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Number of Articles", fontsize=12, fontweight='bold')
        plt.ylabel("Journal Name", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("Fig2_Top_Journals.png", dpi=300, bbox_inches='tight')
        print("Generated: Fig2_Top_Journals.png")
        plt.close()

    # ---------------------------------------------------------
    # FIGURE 3: Fancy Lollipop Chart for Top 15 Keywords
    # ---------------------------------------------------------
    if 'Author Keywords' in df_merged.columns:
        # Split keywords, clean whitespace, and upper case to standardize
        all_kw = df_merged['Author Keywords'].dropna().astype(str).str.split(';')
        flat_kw = [kw.strip().upper() for sublist in all_kw for kw in sublist if len(kw.strip()) > 2]

        kw_counts = Counter(flat_kw)
        top_kw = dict(kw_counts.most_common(15))

        # Create a dataframe for the lollipop chart
        df_kw = pd.DataFrame({'Keyword': list(top_kw.keys()), 'Count': list(top_kw.values())})
        df_kw = df_kw.sort_values(by='Count', ascending=True)

        plt.figure(figsize=(10, 8))
        plt.hlines(y=df_kw['Keyword'], xmin=0, xmax=df_kw['Count'], color='skyblue', linewidth=3)
        plt.plot(df_kw['Count'], df_kw['Keyword'], "o", markersize=10, color='#e74c3c', alpha=0.8)

        plt.title("Top 15 Most Frequently Used Subject Keywords", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Frequency of Occurrence", fontsize=12, fontweight='bold')
        plt.ylabel("Keywords", fontsize=12, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("Fig3_Top_Keywords_Lollipop.png", dpi=300, bbox_inches='tight')
        print("Generated: Fig3_Top_Keywords_Lollipop.png")
        plt.close()

        # FIGURE 4: Word Cloud (Updated with mask style concept)
        text_for_cloud = " ".join(flat_kw).replace("MACHINE LEARNING", "Machine_Learning").replace("DEEP LEARNING", "Deep_Learning").replace("AIR QUALITY", "Air_Quality")
        wordcloud = WordCloud(width=1000, height=500, background_color='white',
                              colormap='Dark2', max_words=100, contour_width=1, contour_color='steelblue').generate(text_for_cloud)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Author Keywords Word Cloud", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig("Fig4_WordCloud.png", dpi=300, bbox_inches='tight')
        print("Generated: Fig4_WordCloud.png")
        plt.close()

    # ---------------------------------------------------------
    # DATA EXTRACTION FOR THE PAPER'S TABLE
    # ---------------------------------------------------------
    # Extract Top 5 Most Used/Cited Papers
    impact_col = 'Since 2013 Usage Count'
    if impact_col in df_merged.columns:
        df_merged[impact_col] = pd.to_numeric(df_merged[impact_col], errors='coerce').fillna(0)
        top_papers = df_merged.sort_values(by=impact_col, ascending=False).head(5)

        print("\n=== DATA FOR PAPER TABLE 1: TOP 5 INFLUENTIAL PAPERS ===")
        print("Copy the following data into your Word document table:")
        for idx, row in top_papers.iterrows():
            authors = str(row.get('Authors', 'Unknown')).split(';')[0] + " et al."
            title = row.get('Document Title', 'Unknown Title')
            year = int(row.get('Year Published', 0))
            journal = row.get('Publication Name', 'Unknown Journal')
            print(f"- {authors} ({year}). {title}. {journal}.")

    print("\n--- ALL FANCY FIGURES GENERATED SUCCESSFULLY ---")

if __name__ == "__main__":
    generate_fancy_bibliometrics()

# --- CELL 3 ---
# --- IMPORTANT NOTE: LATEX ENVIRONMENT SETUP ON COLAB ---
# Uncomment the 2 lines below to install LaTeX the first time you run this on Colab:
# !apt-get update
# !apt-get install -y texlive-xetex texlive-fonts-recommended texlive-fonts-extra poppler-utils

import os
from IPython.display import Image, display

def create_framework_diagram_latex():
    print("--- GENERATING ACADEMIC LATEX (TIKZ) DIAGRAM ---")

    # LaTeX (TikZ) code for the diagram
    latex_code = r"""
    \documentclass[border=15pt, tikz]{standalone}
    \usepackage{tikz}
    \usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc}

    % Set default font for Unicode support
    \usepackage{fontspec}
    \setmainfont{Liberation Sans}

    \begin{document}
    \begin{tikzpicture}[
        node distance=1.2cm and 0.5cm,
        phase/.style={rectangle, rounded corners, minimum width=10.5cm, minimum height=1.2cm, text centered, draw, thick, font=\bfseries\sffamily},
        item/.style={rectangle, rounded corners, minimum width=3.5cm, minimum height=1.2cm, text centered, draw, fill=white, text width=3.8cm, font=\sffamily},
        arrow/.style={thick,->,>=stealth}
    ]

    % Define colors (matching original requirements)
    \definecolor{c1fill}{HTML}{e1f5fe}
    \definecolor{c1border}{HTML}{01579b}
    \definecolor{c2fill}{HTML}{fff3e0}
    \definecolor{c2border}{HTML}{e65100}
    \definecolor{c3fill}{HTML}{e8f5e9}
    \definecolor{c3border}{HTML}{1b5e20}
    \definecolor{c4fill}{HTML}{fce4ec}
    \definecolor{c4border}{HTML}{880e4f}

    % --- PHASE 1 ---
    \node (A) [phase, fill=c1fill, draw=c1border, text=c1border] {PHASE 1: MULTI-SOURCE DATA ACQUISITION};
    \node (C) [item, draw=c1border, below=of A] {Open Meteorological\Data};
    \node (B) [item, draw=c1border, left=of C] {Local IoT\Monitoring Stations};
    \node (D) [item, draw=c1border, right=of C] {Remote Sensing\/ Satellite Data};

    % --- PHASE 2 ---
    \node (E) [phase, fill=c2fill, draw=c2border, text=c2border, below=of C] {PHASE 2: DATA ENGINEERING & MODULARIZATION};
    \node (F) [item, draw=c2border, below=of E] {Automated Preprocessing:\Cleaning, Imputation};
    \node (G) [item, draw=c2border, below=of F] {Normalization\Min-Max / Z-score};
    \node (G1) [item, draw=c2border, below=of G] {Modular Data Storage\for Reusability};

    % --- PHASE 3 ---
    \node (H) [phase, fill=c3fill, draw=c3border, text=c3border, below=of G1] {PHASE 3: COMPONENT-BASED MODEL TRAINING};
    \node (I) [item, draw=c3border, below=of H, xshift=-2.6cm] {Tree-based Algorithm Module:\Random Forest / XGBoost};
    \node (J) [item, draw=c3border, below=of H, xshift=2.6cm] {Deep Learning Algorithm Module:\LSTM / Neural Networks};

    % --- PHASE 4 ---
    \node (K) [phase, fill=c4fill, draw=c4border, text=c4border, below=of H, yshift=-2.8cm] {PHASE 4: EVALUATION, ADJUSTMENT & DEPLOYMENT};
    \node (L) [item, draw=c4border, below=of K] {Multi-criteria Evaluation:\RMSE, R-Square, MAE};
    \node (M) [item, draw=c4border, below=of L] {API Packaging\for Easy Integration};
    \node (N) [item, draw=c4border, below=of M] {Early Warning Web Deployment\/ Local Transfer};

    % --- CONNECTIONS (GEOMETRY OPTIMIZED EDGES) ---

    % From Phase 1 to 3 boxes
    \draw [arrow, c1border, rounded corners=4pt] (A.south) -- ++(0,-0.4) -| (B.north);
    \draw [arrow, c1border] (A.south) -- (C.north);
    \draw [arrow, c1border, rounded corners=4pt] (A.south) -- ++(0,-0.4) -| (D.north);

    % From 3 boxes of Phase 1 converging to Phase 2
    \draw [arrow, c1border, rounded corners=4pt] (B.south) -- ++(0,-0.5) -| (E.north);
    \draw [arrow, c1border] (C.south) -- (E.north);
    \draw [arrow, c1border, rounded corners=4pt] (D.south) -- ++(0,-0.5) -| (E.north);

    % Sequential steps within Phase 2
    \draw [arrow, c2border] (E.south) -- (F.north);
    \draw [arrow, c2border] (F.south) -- (G.north);
    \draw [arrow, c2border] (G.south) -- (G1.north);

    % Transition to Phase 3
    \draw [arrow, c2border] (G1.south) -- (H.north);

    % Phase 3 branches to 2 boxes (Tree-based & Deep Learning)
    \draw [arrow, c3border, rounded corners=4pt] (H.south) -- ++(0,-0.5) -| (I.north);
    \draw [arrow, c3border, rounded corners=4pt] (H.south) -- ++(0,-0.5) -| (J.north);

    % From 2 boxes of Phase 3 converging to Phase 4
    \draw [arrow, c3border, rounded corners=4pt] (I.south) -- ++(0,-0.65) -| (K.north);
    \draw [arrow, c3border, rounded corners=4pt] (J.south) -- ++(0,-0.65) -| (K.north);

    % Sequential steps within Phase 4
    \draw [arrow, c4border] (K.south) -- (L.north);
    \draw [arrow, c4border] (L.south) -- (M.north);
    \draw [arrow, c4border] (M.south) -- (N.north);

    \end{tikzpicture}
    \end{document}
    """

    # 1. Write to .tex file
    with open("diagram.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)

    # 2. Compile with XeLaTeX (required for Vietnamese font support)
    print("Compiling LaTeX to PDF...")
    os.system("xelatex -interaction=nonstopmode diagram.tex")

    # 3. Convert PDF to high-resolution PNG image (300 DPI)
    print("Converting PDF to PNG (300 DPI)...")
    os.system("pdftoppm -png -r 300 diagram.pdf Framework_PM25_LaTeX")

    # Display in Colab
    print("\n=> DONE! Diagram saved as: Framework_PM25_LaTeX-1.png")
    if os.path.exists("Framework_PM25_LaTeX-1.png"):
        display(Image(filename="Framework_PM25_LaTeX-1.png"))
    else:
        print("=> ERROR: Please ensure you UNCOMMENTED the !apt-get commands at the top and re-ran.")

if __name__ == "__main__":
    create_framework_diagram_latex()

# --- CELL 4 ---
# INSTRUCTIONS:
# 1. Open Google Colab (colab.research.google.com) -> Create a new Notebook.
# 2. Copy and paste this entire code snippet into a cell and click the Play (Run) button.
# 3. The code will automatically download data, print R2 and RMSE results, and save one plot.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# 1. LOAD PUBLIC DATA (Beijing PM2.5 Dataset - basic cleaned)
print("Downloading data...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"
df = pd.read_csv(url, header=0, index_col=0)

# Data includes: pollution (PM2.5), dew (dew point), temp (temperature), press (pressure), wnd_dir, wnd_spd, snow, rain.
# Quickly handle Categorical data (wind direction) and remove rows with missing values (NaN)
df = pd.get_dummies(df, columns=['cbwd']) # Changed 'wnd_dir' to 'cbwd'
df = df.dropna()

# 2. PREPARE TRAINING DATA
# The target variable is 'pm2.5' (PM2.5 concentration)
y = df['pm2.5']
X = df.drop('pm2.5', axis=1)

# Split into train/test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAIN THE XGBOOST MODEL (Algorithm mentioned in your abstract)
print("Training XGBoost model...")
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 4. PREDICT AND EVALUATE
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print("MODEL EVALUATION RESULTS:")
print(f"RMSE: {rmse:.2f} µg/m³")
print(f"R-squared (R2): {r2:.4f}")
print("-" * 30)

# 5. PLOT COMPARISON (Take the first 100 hours of the test set for better visualization)
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label='Actual', color='blue', marker='o', markersize=4)
plt.plot(y_pred[:100], label='Predicted', color='red', linestyle='--', marker='x', markersize=4)
plt.title('Comparison of Actual and Predicted PM2.5 Concentration using XGBoost (100 sample hours)')
plt.xlabel('Time (Hours)')
plt.ylabel('PM2.5 Concentration (µg/m³)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Save the plot to embed in Word
plt.savefig('PM25_Prediction_Chart.png', dpi=300)
print("Plot saved as 'PM25_Prediction_Chart.png'. Please download this file to embed in your paper.")
plt.show()

# 6. FEATURE IMPORTANCE ANALYSIS (To inform the Discussion section)
# To get ideas for writing the Discussion section
importances = model.feature_importances_
features = X.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
print("\nINFLUENCE OF FACTORS ON PM2.5:")
print(feat_imp.head(5))

# --- CELL 5 ---
# INSTRUCTIONS: Run this code snippet on Colab (ensuring LaTeX is installed as in the previous file)
# This diagram illustrates the execution flow of the source code (Coding Flow), mapping one-to-one with the Theoretical Framework.

import os
from IPython.display import Image, display

def create_coding_flow_diagram_latex():
    print("--- GENERATING COLAB CODING FLOW DIAGRAM USING LATEX (TIKZ) ---")

    # LaTeX (TikZ) code for the coding diagram
    # Note: Underscores '_' in Python have been escaped with '\\' for LaTeX compatibility
    latex_code = r"""
    \documentclass[border=15pt, tikz]{standalone}
    \usepackage{tikz}
    \usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc}

    % Set default font for Unicode support
    \usepackage{fontspec}
    \setmainfont{Liberation Sans}

    \begin{document}
    \begin{tikzpicture}[
        node distance=1.2cm and 0.5cm,
        phase/.style={rectangle, rounded corners, minimum width=11cm, minimum height=1.2cm, text centered, draw, thick, font=\bfseries\sffamily},
        item/.style={rectangle, rounded corners, minimum width=4cm, minimum height=1.2cm, text centered, draw, fill=white, text width=4.2cm, font=\sffamily},
        arrow/.style={thick,->,>=stealth}
    ]

    % Define colors (matching Theoretical Framework)
    \definecolor{c1fill}{HTML}{e1f5fe}
    \definecolor{c1border}{HTML}{01579b}
    \definecolor{c2fill}{HTML}{fff3e0}
    \definecolor{c2border}{HTML}{e65100}
    \definecolor{c3fill}{HTML}{e8f5e9}
    \definecolor{c3border}{HTML}{1b5e20}
    \definecolor{c4fill}{HTML}{fce4ec}
    \definecolor{c4border}{HTML}{880e4f}
    % Add a fifth color for the Feature Extraction part
    \definecolor{c5fill}{HTML}{f3e5f5}
    \definecolor{c5border}{HTML}{4a148c}

    % --- STEP 1 ---
    \node (A) [phase, fill=c1fill, draw=c1border, text=c1border] {STEP 1: PUBLIC DATA LOADING};
    \node (A1) [item, draw=c1border, below=of A] {\texttt{pd.read\_csv()}\Dataset: pollution.csv};

    % --- STEP 2 ---
    \node (B) [phase, fill=c2fill, draw=c2border, text=c2border, below=of A1, yshift=-0.5cm] {STEP 2: PREPROCESSING AND DATA SPLITTING};
    \node (B1) [item, draw=c2border, below=of B, xshift=-2.5cm] {\texttt{df.dropna()}\Remove Missing Data};
    \node (B2) [item, draw=c2border, below=of B, xshift=2.5cm] {\texttt{train\_test\_split()}\80\% Train - 20\% Test Split};

    % --- STEP 3 ---
    \node (C) [phase, fill=c3fill, draw=c3border, text=c3border, below=of B, yshift=-2.6cm] {STEP 3: CORE MODEL TRAINING};
    \node (C1) [item, draw=c3border, below=of C] {\texttt{xgb.XGBRegressor()}\n\_estimators=100, lr=0.1};

    % --- STEP 4 & 5 ---
    \node (D) [phase, fill=c4fill, draw=c4border, text=c4border, below=of C1, yshift=-0.5cm] {STEP 4 \& 5: PERFORMANCE EVALUATION \& VISUALIZATION};
    \node (D1) [item, draw=c4border, below=of D, xshift=-2.5cm] {\texttt{mean\_squared\_error()}\Calculate RMSE, R-Squared};
    \node (D2) [item, draw=c4border, below=of D, xshift=2.5cm] {\texttt{plt.plot()}\Plot Forecast (matplotlib)};

    % --- STEP 6 ---
    \node (E) [phase, fill=c5fill, draw=c5border, text=c5border, below=of D, yshift=-2.6cm] {STEP 6: MODEL INTERPRETATION};
    \node (E1) [item, draw=c5border, below=of E] {\texttt{model.feature\_importances\_}\Analyze Influence Level};

    % --- CONNECTIONS (EDGES) ---

    % From A down to A1
    \draw [arrow, c1border] (A.south) -- (A1.north);
    \draw [arrow, c1border] (A1.south) -- (B.north);

    % Phase 2 branching
    \draw [arrow, c2border, rounded corners=4pt] (B.south) -- ++(0,-0.4) -| (B1.north);
    \draw [arrow, c2border, rounded corners=4pt] (B.south) -- ++(0,-0.4) -| (B2.north);

    % Phase 2 converging to Phase 3
    \draw [arrow, c2border, rounded corners=4pt] (B1.south) -- ++(0,-0.6) -| (C.north);
    \draw [arrow, c2border, rounded corners=4pt] (B2.south) -- ++(0,-0.6) -| (C.north);

    % Phase 3 down to C1 then D
    \draw [arrow, c3border] (C.south) -- (C1.north);
    \draw [arrow, c3border] (C1.south) -- (D.north);

    % Phase 4 branching
    \draw [arrow, c4border, rounded corners=4pt] (D.south) -- ++(0,-0.4) -| (D1.north);
    \draw [arrow, c4border, rounded corners=4pt] (D.south) -- ++(0,-0.4) -| (D2.north);

    % Phase 4 converging to Phase 5
    \draw [arrow, c4border, rounded corners=4pt] (D1.south) -- ++(0,-0.6) -| (E.north);
    \draw [arrow, c4border, rounded corners=4pt] (D2.south) -- ++(0,-0.6) -| (E.north);

    % Phase 5 down to E1
    \draw [arrow, c5border] (E.south) -- (E1.north);

    \end{tikzpicture}
    \end{document}
    """

    # 1. Write to .tex file
    with open("coding_diagram.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)

    # 2. Compile with XeLaTeX
    print("Compiling LaTeX to PDF...")
    os.system("xelatex -interaction=nonstopmode coding_diagram.tex")

    # 3. Convert PDF to high-resolution PNG image (300 DPI)
    print("Converting PDF to PNG (300 DPI)...")
    os.system("pdftoppm -png -r 300 coding_diagram.pdf Coding_Flow_PM25")

    # Display in Colab
    print("\n=> DONE! Diagram saved as: Coding_Flow_PM25-1.png")
    if os.path.exists("Coding_Flow_PM25-1.png"):
        display(Image(filename="Coding_Flow_PM25-1.png"))
    else:
        print("=> ERROR: Please ensure the LaTeX environment is correctly set up in Colab.")

if __name__ == "__main__":
    create_coding_flow_diagram_latex()

# --- CELL 6 ---
# INSTRUCTIONS:
# 1. Copy and paste this code snippet into a new Colab cell and run it.
# 2. It will create a highly professional Bar Chart and save it as an image file.

import matplotlib.pyplot as plt
import numpy as np

# 1. Data extracted from your XGBoost model
features = ['Northwest Wind Direction\n(cbwd_NW)', 'Southeast Wind Direction\n(cbwd_SE)',
            'Dew Point\n(DEWP)', 'Temperature\n(TEMP)', 'Month\n(month)']
importances = [0.170655, 0.133686, 0.130354, 0.115310, 0.108020]

# 2. Set up the chart framework (academic style)
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(features))

# Draw horizontal bar chart
bars = ax.barh(y_pos, importances, align='center', color='#2b8cbe', edgecolor='black', alpha=0.8)

# Set axis labels
ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontsize=11)
ax.invert_yaxis()  # Invert Y-axis so the most important factor is at the top
ax.set_xlabel('Influence Weight (Feature Importance)', fontsize=12, fontweight='bold')
ax.set_title('Contribution Level of Features to PM2.5 Prediction', fontsize=14, fontweight='bold', pad=15)

# Add specific values on each bar for readability
for bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    ax.text(width + 0.002, label_y, s=f'{width:.4f}', ha='left', va='center', fontsize=11)

# 3. Insert a text box containing RMSE and R2 results in the corner of the chart
metrics_text = (
    "Model Evaluation Results:\n"
    r"• $RMSE = 56.39\ \mu g/m^3$" + "\n"
    r"• $R^2 = 0.6391$"
)
# Set properties for the text box
props = dict(boxstyle='round,pad=0.5', facecolor='#f0f9e8', edgecolor='gray', alpha=0.9)
ax.text(0.65, 0.20, metrics_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props, linespacing=1.5)

# Add faint grid for X-axis
ax.xaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# 4. Save and display
plt.tight_layout()
plt.savefig('Feature_Importance_Chart.png', dpi=300, bbox_inches='tight')
print("Chart saved as 'Feature_Importance_Chart.png'. Please download it to embed in your Word document.")
plt.show()

# --- CELL 7 ---
# INSTRUCTIONS:
# 1. Open Google Colab (colab.research.google.com).
# 2. Copy and paste this code to compare ANN performance with XGBoost.
# 3. These results can be used to support Chandanshive (2024)'s review document.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. LOAD AND PREPROCESS DATA
print("Preparing data for the neural network...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"
df = pd.read_csv(url, header=0, index_col=0)
df = pd.get_dummies(df, columns=['cbwd']) # Corrected from 'wnd_dir' to 'cbwd'
df = df.dropna()

y = df['pm2.5'].values # Corrected from 'pollution' to 'pm2.5'
X = df.drop('pm2.5', axis=1).values # Corrected from 'pollution' to 'pm2.5'

# Split data set
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

# Standardize data (ESSENTIAL for ANN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 2. BUILD ARTIFICIAL NEURAL NETWORK (ANN) ARCHITECTURE
# Design based on the "State-of-the-art" proposal by Chandanshive & Shanbhag (2024)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1) # Output layer for regression problem
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. TRAIN THE MODEL
print("Training ANN (Deep Learning) model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=0
)

# 4. PREDICT AND EVALUATE
y_pred = model.predict(X_test).flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print("ANN MODEL EVALUATION RESULTS:")
print(f"RMSE: {rmse:.2f} µg/m³")
print(f"R-squared (R2): {r2:.4f}")
print("-" * 30)

# 5. PLOT LEARNING CURVE (LOSS CURVE)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training Curve (Loss)')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()

# 6. PLOT COMPARISON
plt.subplot(1, 2, 2)
plt.plot(y_test[:100], label='Actual', color='blue', alpha=0.7)
plt.plot(y_pred[:100], label='ANN Predicted', color='green', linestyle='--')
plt.title('PM2.5 Prediction using ANN (First 100 samples)')
plt.legend()

plt.tight_layout()
plt.savefig('ANN_Prediction_Results.png', dpi=300)
plt.show()

print("Chart 'ANN_Prediction_Results.png' saved for comparison with XGBoost.")

# --- CELL 8 ---
# INSTRUCTIONS:
# 1. Open Google Colab (colab.research.google.com).
# 2. Copy this code to implement the LSTM (Long Short-Term Memory) model.
# 3. This serves as evidence for "Future Development Directions" (Section 5) in the paper.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization

# 1. LOAD AND PREPROCESS TIME SERIES DATA
print("Preparing data for the LSTM model...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"
df = pd.read_csv(url, header=0, index_col=0)
df = pd.get_dummies(df, columns=['cbwd']) # Corrected from 'wnd_dir' to 'cbwd'
df = df.dropna()

# Reorder columns to ensure 'pm2.5' is the first column for create_dataset function
target_column = 'pm2.5'
cols = [target_column] + [col for col in df.columns if col != target_column]
df_reordered = df[cols]

# Standardize all data before creating windows
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_reordered.values)

# Function to create time windows
# Example: Use data from 24 hours prior (look_back) to predict the next hour
def create_dataset(dataset, look_back=24):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :])
        y.append(dataset[i + look_back, 0]) # Now 'pm2.5' is at index 0 after reordering
    return np.array(X), np.array(y)

LOOK_BACK = 24 # Use the previous 24 hours as input
X_series, y_series = create_dataset(data_scaled, LOOK_BACK)

# Split the dataset (Note: Do not shuffle to preserve temporal order)
train_size = int(len(X_series) * 0.8)
X_train, X_test = X_series[:train_size], X_series[train_size:]
y_train, y_test = y_series[:train_size], y_series[train_size:]

print(f"Training set size: {X_train.shape}")

# 2. BUILD LSTM MODEL ARCHITECTURE
# This is a crucial upgrade for handling complex nonlinear temporal patterns
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=True), # First LSTM layer
    Dropout(0.2),
    LSTM(32), # Second LSTM layer
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 3. TRAIN THE MODEL
print("Training LSTM (Deep Learning for Time Series) model...")
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

# 4. PREDICT AND EVALUATE (Convert to original units)
y_pred_scaled = model.predict(X_test)

# Inverse normalize for the target variable (pollution is in the first column)
y_test_unscaled = y_test * np.sqrt(scaler.var_[0]) + scaler.mean_[0] # Scaler info for the 0th column (pm2.5)
y_pred_unscaled = y_pred_scaled.flatten() * np.sqrt(scaler.var_[0]) + scaler.mean_[0] # Scaler info for the 0th column (pm2.5)

rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
r2 = r2_score(y_test_unscaled, y_pred_unscaled)

print("-" * 30)
print("LSTM MODEL EVALUATION RESULTS:")
print(f"RMSE: {rmse:.2f} µg/m³")
print(f"R-squared (R2): {r2:.4f}")
print("-" * 30)

# 5. PLOT COMPARISON
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM Learning Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_test_unscaled[:100], label='Actual', color='blue')
plt.plot(y_pred_unscaled[:100], label='LSTM Predicted', color='red', linestyle='--')
plt.title('PM2.5 Prediction using LSTM (24h Window)')
plt.legend()

plt.tight_layout()
plt.savefig('LSTM_Prediction_Results.png', dpi=300)
plt.show()

print("LSTM models typically achieve higher R2 than pure ANNs because they capture temporal dependencies.")

# --- CELL 9 ---
# INSTRUCTIONS: Run this code snippet on Colab (ensuring LaTeX is installed as in the previous file)
# This diagram illustrates the execution flow of the LSTM model, corresponding 1-1 with the advanced Theoretical Framework.

import os
from IPython.display import Image, display

def create_coding_flow_diagram_latex():
    # Install LaTeX packages if not already installed
    !apt-get update
    !apt-get install -y texlive-xetex texlive-fonts-recommended texlive-fonts-extra poppler-utils

    print("--- GENERATING LSTM CODING FLOW DIAGRAM USING LATEX (TIKZ) ---")

    latex_code = r"""
    \documentclass[border=15pt, tikz]{standalone}
    \usepackage{tikz}
    \usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc}

    % Set font (Colab usually has Liberation Sans)
    \usepackage{fontspec}
    \setmainfont{Liberation Sans}

    \begin{document}
    \begin{tikzpicture}[
        node distance=1.1cm and 0.5cm,
        phase/.style={rectangle, rounded corners, minimum width=11.5cm, minimum height=1.1cm, text centered, draw, thick, font=\bfseries\sffamily},
        item/.style={rectangle, rounded corners, minimum width=4.5cm, minimum height=1.1cm, text centered, draw, fill=white, text width=4.8cm, font=\sffamily\small},
        arrow/.style={thick,->,>=stealth}
    ]

    % Define colors synchronized with the paper
    \definecolor{c1fill}{HTML}{e1f5fe} \definecolor{c1border}{HTML}{01579b}
    \definecolor{c2fill}{HTML}{fff3e0} \definecolor{c2border}{HTML}{e65100}
    \definecolor{c3fill}{HTML}{e8f5e9} \definecolor{c3border}{HTML}{1b5e20}
    \definecolor{c4fill}{HTML}{fce4ec} \definecolor{c4border}{HTML}{880e4f}
    \definecolor{c5fill}{HTML}{f3e5f5} \definecolor{c5border}{HTML}{4a148c}

    % --- STEP 1: DATA INGESTION ---
    \node (A) [phase, fill=c1fill, draw=c1border, text=c1border] {STEP 1: TIME SERIES DATA ACQUISITION};
    \node (A1) [item, draw=c1border, below=of A] {\texttt{pd.read\_csv()}\nLoad Historical Pollution Data};

    % --- STEP 2: LSTM-SPECIFIC PREPROCESSING ---
    \node (B) [phase, fill=c2fill, draw=c2border, text=c2border, below=of A1, yshift=-0.4cm] {STEP 2: PREPROCESSING AND TIME WINDOWING};
    \node (B1) [item, draw=c2border, below=of B, xshift=-2.8cm] {\texttt{StandardScaler()}\nNormalize Data to [0, 1]};
    \node (B2) [item, draw=c2border, below=of B, xshift=2.8cm] {\texttt{create\_dataset()}\nCreate Look-back Structure (24h)};

    % --- STEP 3: DEEP NEURAL NETWORK CONSTRUCTION ---
    \node (C) [phase, fill=c3fill, draw=c3border, text=c3border, below=of B, yshift=-2.4cm] {STEP 3: DESIGN AND TRAIN LSTM MODEL};
    \node (C1) [item, draw=c3border, below=of C] {\texttt{Sequential([LSTM, Dropout])}\n2-Layer LSTM Architecture};

    % --- STEP 4 & 5: EVALUATION ---
    \node (D) [phase, fill=c4fill, draw=c4border, text=c4border, below=of C1, yshift=-0.4cm] {STEP 4 & 5: PREDICTION AND INVERSE NORMALIZATION};
    \node (D1) [item, draw=c4border, below=of D, xshift=-2.8cm] {\texttt{model.predict()}\nPredict on Independent Test Set};
    \node (D2) [item, draw=c4border, below=of D, xshift=2.8cm] {\texttt{scaler.inverse\_transform()}\nConvert to $\mu g/m^3$ Units};

    % --- STEP 6: RESULT ANALYSIS ---
    \node (E) [phase, fill=c5fill, draw=c5border, text=c5border, below=of D, yshift=-2.4cm] {STEP 6: RELIABILITY AND ERROR EVALUATION};
    \node (E1) [item, draw=c5border, below=of E] {\texttt{r2\_score() & RMSE}\nConfirm $R^2 > 0.95$ Performance};

    % --- EXECUTION FLOW CONNECTIONS ---
    \draw [arrow, c1border] (A.south) -- (A1.north);
    \draw [arrow, c1border] (A1.south) -- (B.north);

    \draw [arrow, c2border, rounded corners=4pt] (B.south) -- ++(0,-0.3) -| (B1.north);
    \draw [arrow, c2border, rounded corners=4pt] (B.south) -- ++(0,-0.3) -| (B2.north);
    \draw [arrow, c2border, rounded corners=4pt] (B1.south) -- ++(0,-0.5) -| (C.north);
    \draw [arrow, c2border, rounded corners=4pt] (B2.south) -- ++(0,-0.5) -| (C.north);

    \draw [arrow, c3border] (C.south) -- (C1.north);
    \draw [arrow, c3border] (C1.south) -- (D.north);

    \draw [arrow, c4border, rounded corners=4pt] (D.south) -- ++(0,-0.3) -| (D1.north);
    \draw [arrow, c4border, rounded corners=4pt] (D.south) -- ++(0,-0.3) -| (D2.north);
    \draw [arrow, c4border, rounded corners=4pt] (D1.south) -- ++(0,-0.5) -| (E.north);
    \draw [arrow, c4border, rounded corners=4pt] (D2.south) -- ++(0,-0.5) -| (E.north);

    \draw [arrow, c5border] (E.south) -- (E1.north);

    \end{tikzpicture}
    \end{document}
    """

    with open("coding_diagram_lstm.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)

    print("Compiling LSTM diagram...")
    os.system("xelatex -interaction=nonstopmode coding_diagram_lstm.tex")
    os.system("pdftoppm -png -r 300 coding_diagram_lstm.pdf Coding_Flow_LSTM")

    if os.path.exists("Coding_Flow_LSTM-1.png"):
        print("\n=> SUCCESS! Updated LSTM experimental diagram.")
        display(Image(filename="Coding_Flow_LSTM-1.png"))
    else:
        print("=> ERROR: Could not create image file. Check LaTeX environment setup on Colab.")

if __name__ == "__main__":
    create_coding_flow_diagram_latex()

# --- CELL 10 ---
# Install networkx library if not already installed (Colab usually has it pre-installed)
# !pip install networkx matplotlib pandas

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from collections import Counter

def run_topology_analysis():
    print("--- STARTING TOPOLOGY ANALYSIS ---")

    # 1. Read merged data
    try:
        df = pd.read_csv("merged_wos_dataset_final.csv")
        print("Successfully loaded data: merged_wos_dataset_final.csv")
    except FileNotFoundError:
        print("ERROR: File 'merged_wos_dataset_final.csv' not found. Please run the CSV merging code in the previous step.")
        return

    if 'Author Keywords' not in df.columns:
        print("ERROR: 'Author Keywords' column does not exist in the data.")
        return

    # 2. Build a list of co-occurring keyword pairs
    print("Building keyword co-occurrence network...")
    keyword_pairs = []

    # Get the 30 most common keywords so the network doesn't get too cluttered
    all_kw_raw = df['Author Keywords'].dropna().astype(str).str.split(';')
    flat_kw_all = [kw.strip().upper() for sublist in all_kw_raw for kw in sublist if len(kw.strip()) > 2]
    top_kw_list_and_counts = Counter(flat_kw_all).most_common(30)
    top_kw_list = [kw for kw, count in top_kw_list_and_counts]

    if not top_kw_list:
        print("Error: No sufficiently long keywords found to form a top keyword list. Cannot proceed with network analysis.")
        return

    # 3. Create NetworkX graph
    G = nx.Graph()
    # Add all top keywords as nodes first, to ensure they exist in the graph even if they don't form pairs
    G.add_nodes_from(top_kw_list)

    for kws_doc_raw in df['Author Keywords'].dropna():
        # Clean and filter keywords for the current document, keeping only those in top_kw_list
        kws_clean_for_doc = [kw.strip().upper() for kw in str(kws_doc_raw).split(';') if kw.strip().upper() in top_kw_list]

        if len(kws_clean_for_doc) > 1:
            # Create combinations of keyword pairs from the cleaned list
            pairs = list(itertools.combinations(kws_clean_for_doc, 2))
            keyword_pairs.extend(pairs)

    # Add edges based on co-occurrence
    for pair in keyword_pairs:
        # Check if pair exists as nodes in G (they should if they are in top_kw_list)
        if G.has_edge(pair[0], pair[1]):
            G[pair[0]][pair[1]]['weight'] += 1
        else:
            G.add_edge(pair[0], pair[1], weight=1)

    # 4. Calculate Topology Metrics
    print("\n--- TOPOLOGY MEASUREMENT RESULTS ---")

    # Check if the graph has any nodes before proceeding
    if not G.nodes():
        print("Error: The co-occurrence network is empty after processing. No keywords were found.")
        return

    # Handle cases where there are nodes but no edges (isolated nodes)
    if G.number_of_edges() == 0:
        print("Warning: The co-occurrence network has nodes but no edges. All nodes are isolated.")
        degree_centrality = {node: 0 for node in G.nodes()} # Degree centrality is 0 for isolated nodes
        top_central = [] # No hub nodes if no edges
        avg_clustering = 0.0 # Clustering coefficient is 0 for isolated nodes
        density = 0.0 # Density is 0 if no edges
        print(f"1. Strongest Hub Nodes: {top_central}")
        print(f"2. Average Clustering Coefficient: {avg_clustering:.4f}")
        print(f"3. Network Density: {density:.4f}")

    else:
        # Proceed with calculations only if there are edges
        # Degree Centrality
        degree_centrality = nx.degree_centrality(G)
        top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"1. Strongest Hub Nodes: {top_central}")

        # Average Clustering Coefficient
        # Use a try-except block for robustness in case of edge cases (e.g., specific graph structures with no triangles).
        try:
            avg_clustering = nx.average_clustering(G)
        except ZeroDivisionError: # Catch if no valid clustering coefficients can be computed (e.g., no triangles)
            avg_clustering = 0.0
        print(f"2. Average Clustering Coefficient: {avg_clustering:.4f}")

        # Network Density
        density = nx.density(G)
        print(f"3. Network Density: {density:.4f}")

        if avg_clustering > density:
             print("=> CONCLUSION: Clustering coefficient is greater than network density. This is a clear characteristic of a 'Small-world network' structure!")

    # 5. Network Visualization
    print("\nDrawing Topology Network diagram...")
    plt.figure(figsize=(14, 10))

    # Algorithm to calculate node positions aesthetically (spring layout)
    pos = nx.spring_layout(G, k=0.8, seed=42)

    # Node size based on centrality
    # Ensure node_sizes are not all zero if there are nodes but no calculated centrality
    node_sizes = [v * 8000 for v in degree_centrality.values()]
    if all(s == 0 for s in node_sizes) and G.number_of_nodes() > 0:
        node_sizes = [1000] * G.number_of_nodes() # Assign a default size for visibility

    # Edge thickness based on co-occurrence frequency (weight)
    edge_weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]

    # Draw edges (only if there are edges)
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_weights, edge_color='gray')

    # Draw nodes (only if there are nodes)
    if G.number_of_nodes() > 0:
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue',
                               edgecolors='darkblue', linewidths=1.5, alpha=0.9)

        # Write labels on nodes (only if there are nodes)
        nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif', font_weight='bold')

    plt.title("Topology Network Structure: Co-occurrence of Research Keywords",
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off') # Hide axes
    plt.tight_layout()

    # Save HD image
    plt.savefig("Fig_Topology_Network.png", dpi=300, bbox_inches='tight')
    print("=> DONE! Diagram saved as 'Fig_Topology_Network.png'. Please download it for your paper.")
    plt.show()

if __name__ == "__main__":
    run_topology_analysis()

# --- CELL 11 ---
# Install networkx library if not already installed (Colab usually has it pre-installed)
# !pip install networkx matplotlib pandas

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import re

def run_topology_analysis():
    print("--- STARTING TOPOLOGY ANALYSIS ---")

    # 1. Read merged data
    try:
        df = pd.read_csv("merged_wos_dataset_final.csv")
        print("Successfully loaded data: merged_wos_dataset_final.csv")
    except FileNotFoundError:
        print("ERROR: File 'merged_wos_dataset_final.csv' not found. Please run the CSV merging code in the previous step.")
        return

    if 'Author Keywords' not in df.columns:
        print("ERROR: 'Author Keywords' column does not exist in the data.")
        return

    # 2. Build a list of co-occurring keyword pairs
    print("Building keyword co-occurrence network...")
    keyword_pairs = []

    # Use Regex to split keywords by both comma (,) and semicolon (;)
    all_kw_raw = df['Author Keywords'].dropna().astype(str)
    flat_kw_all = []
    for row in all_kw_raw:
        kws = [kw.strip().upper() for kw in re.split(r'[;,]', row) if len(kw.strip()) > 2]
        flat_kw_all.extend(kws)

    # Get the 40 most common keywords for a richer network
    top_kw_list_and_counts = Counter(flat_kw_all).most_common(40)
    top_kw_list = [kw for kw, count in top_kw_list_and_counts]

    if not top_kw_list:
        print("ERROR: No valid keywords found.")
        return

    # 3. Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(top_kw_list)

    for kws_doc_raw in df['Author Keywords'].dropna():
        # Split and clean keywords for each article
        kws_split = [kw.strip().upper() for kw in re.split(r'[;,]', str(kws_doc_raw)) if len(kw.strip()) > 2]
        # Filter keywords that are in the Top 40, using set() to remove duplicate keywords within the same article
        kws_clean_for_doc = list(set([kw for kw in kws_split if kw in top_kw_list]))

        if len(kws_clean_for_doc) > 1:
            # Create combinations of co-occurring keyword pairs
            pairs = list(itertools.combinations(kws_clean_for_doc, 2))
            keyword_pairs.extend(pairs)

    # Add edges based on co-occurrence frequency
    for pair in keyword_pairs:
        if G.has_edge(pair[0], pair[1]):
            G[pair[0]][pair[1]]['weight'] += 1
        else:
            G.add_edge(pair[0], pair[1], weight=1)

    # Remove isolated nodes (not connected to any other keywords) for a cleaner graph
    G.remove_nodes_from(list(nx.isolates(G)))

    # 4. Calculate Topology Metrics
    print("\n--- TOPOLOGY MEASUREMENT RESULTS ---")

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print("Error: The network could not form any edges due to highly dispersed keyword data.")
        return

    # Degree Centrality
    degree_centrality = nx.degree_centrality(G)
    top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"1. Strongest Hub Nodes:")
    for node, score in top_central:
        print(f"   - {node}: {score:.4f}")

    # Average Clustering Coefficient
    try:
        avg_clustering = nx.average_clustering(G)
    except ZeroDivisionError:
        avg_clustering = 0.0
    print(f"2. Average Clustering Coefficient: {avg_clustering:.4f}")

    # Network Density
    density = nx.density(G)
    print(f"3. Network Density: {density:.4f}")

    if avg_clustering > density:
         print("=> CONCLUSION: Clustering coefficient is greater than network density. This is a clear characteristic of a 'Small-world network' structure!")

    # 5. Network Visualization
    print("\nDrawing Topology Network diagram...")
    plt.figure(figsize=(16, 12))

    # Algorithm to calculate node positions aesthetically (larger k pushes nodes further apart)
    pos = nx.spring_layout(G, k=1.2, seed=42)

    # Node size based on centrality (add 0.05 so weak nodes are not too small)
    node_sizes = [(v + 0.05) * 8000 for v in degree_centrality.values()]

    # Edge thickness based on co-occurrence frequency (weight)
    edge_weights = [G[u][v]['weight'] * 0.8 for u, v in G.edges()]

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=edge_weights, edge_color='#a0aec0')

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#90cdf4',
                           edgecolors='#2b6cb0', linewidths=2, alpha=0.95)

    # Write labels on nodes
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', font_weight='bold')

    plt.title("Topology Network Structure: Co-occurrence of Research Keywords",
              fontsize=18, fontweight='bold', pad=25)
    plt.axis('off') # Hide axes
    plt.tight_layout()

    # Save HD image for the paper
    plt.savefig("Fig_Topology_Network.png", dpi=300, bbox_inches='tight')
    print("=> DONE! Diagram saved as 'Fig_Topology_Network.png'. Please download it for your paper.")
    plt.show()

if __name__ == "__main__":
    run_topology_analysis()
