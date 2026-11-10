# 📊 SEO CTR Prediction & Opportunity Finder

A powerful and customizable Python script for SEO professionals to analyze Google Search Console (GSC) data. It calculates the **Expected CTR** for queries based on search intent, ranking position, and word count, and identifies "Golden Opportunities" — pages with high impressions that severely underperform in clicks.

## 🚀 Features

*   **CLI Support:** Easy to run from the command line with custom input/output paths.
*   **Data Cleaning:** Handles Persian/Arabic numerals, normalizes characters, and removes invalid rows.
*   **Search Intent Classification:** Categorizes queries into Commercial, Navigational, Local, Informational, or Generic. *(You can easily customize the regex patterns in the code for other languages!)*
*   **Multi-tier Expected CTR Model:** Uses a robust fallback system (Intent + Position + Word Count -> Intent + Position -> Position Only) to predict CTR.
*   **Golden Opportunities:** Finds queries with high impressions (default: 1000+) and the lowest CTR gap.
*   **Excel Reporting:** Exports a structured multi-sheet Excel file.

## 🛠️ Installation

1. Clone this repository or download the files.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
