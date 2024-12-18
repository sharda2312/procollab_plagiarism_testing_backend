# Project Plagiarism Checker

## Overview
This project is a **Plagiarism Checker** designed to assess the originality of project submissions. It compares user-provided project titles and descriptions against a database of previously submitted projects, using natural language processing (NLP) techniques to identify potential plagiarism. The system promotes academic integrity and fairness by flagging submissions that exceed a predefined similarity threshold.

## Features
- **Input Processing**: Tokenization, stemming, and stopword removal to clean and prepare text.
- **Database Integration**: Stores and queries project data using MongoDB.
- **Similarity Analysis**: Uses **TF-IDF vectorization** and **cosine similarity** to calculate text similarity.
- **Plagiarism Detection**: Flags submissions with a similarity score above 70%.
- **Real-Time Feedback**: Provides immediate results to users regarding their submission's originality.

## Technology Stack
- **Programming Language**: Python
- **Backend Framework**: Flask
- **Database**: MongoDB
- **NLP Libraries**: NLTK (Natural Language Toolkit)
- **Vectorization**: TF-IDF

## How It Works
1. **User Input**:
   - Users submit a project title and description through the web interface.
2. **Text Preprocessing**:
   - Tokenization, stemming, and removal of stopwords prepare the text for analysis.
3. **Database Query**:
   - MongoDB is queried to fetch relevant data for comparison.
4. **Cosine Similarity Calculation**:
   - TF-IDF vectorization is used to compute similarity scores between the input and database entries.
5. **Plagiarism Threshold**:
   - Submissions with similarity scores above 70% are flagged as potential plagiarism.
6. **Response to User**:
   - Feedback is provided in real-time, either accepting or rejecting the submission based on its similarity score.

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/plagiarism-checker.git
   cd plagiarism-checker
## Installation and Setup

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python app.py
![deployment image](https://drive.google.com/uc?export=view&id=1E2x3i5avdeoRCBmA2zsmj_P_iXyT7oDv)
