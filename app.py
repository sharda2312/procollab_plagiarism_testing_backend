from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymongo

MONGODB_CONNECTION_STRING = "{{shared.MONGODB_CONNECTION_STRING}}"

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data (specific resources) from the server
nltk.downloader.download('punkt')  # Download tokenization data
nltk.downloader.download('stopwords')  # Download stopwords data


# Initialize NLTK and MongoDB connection
client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
db = client["your_database_name"]
collection = db["projects"]

# plagiarism checking code
def check_plagiarism(user_title, user_description):
    
    # Retrieve titles and project descriptions from the database
    project_documents = collection.find({}, {"shortdescription": 1, "description": 1})

    # Tokenization, Stemming, and Stopword Removal
    stemmer = PorterStemmer()

    stop_words = set(stopwords.words('english'))


    def preprocess_descpription(text):
        tokens = word_tokenize(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        filtered_tokens = [token for token in stemmed_tokens if token.lower() not in stop_words]
        return ' '.join([token.lower() for token in filtered_tokens])

    def preprocess_query(text):
        tokens = word_tokenize(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        filtered_tokens = [token for token in stemmed_tokens if token.lower() not in stop_words]
        return ' '.join([token.lower() for token in filtered_tokens])

    # Preprocess user input
    user_title = preprocess_query(user_title)
    user_description = preprocess_query(user_description)

    # Preprocess and vectorize project descriptions from the database
    project_descriptions = [preprocess_descpription(doc["description"]) for doc in project_documents]

    # Calculate cosine similarity
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([user_title + " " + user_description] + project_descriptions)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    plagiarism_score = cosine_similarities
    return plagiarism_score

# API endpoint to check plagiarism
@app.route('/', methods=['POST'])
def api_check_plagiarism():
    try:
        # Get user input from the request
        user_title = request.form.get('title')
        user_description = request.form.get('description')
        
        # Check plagiarism
        plagiarism_score = check_plagiarism(user_title, user_description)
        
        # Determine if the project should be accepted or rejected
        max_similarity_score = max(plagiarism_score[0])
        if max_similarity_score > 0.6 :
            response = {'status': 'rejected', 'message': 'Plagiarism detected! Project rejected.'}
        else:
            response = {'status': 'accepted', 'message': 'Project submitted successfully.'}
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
