from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, Flask!'

# Download NLTK data (specific resources) from the server
nltk.downloader.download('punkt')  # Download tokenization data
nltk.downloader.download('stopwords')  # Download stopwords data

# Function to fetch data from the API and store it globally
global_var = None

def get_data_from_api():
    
    global global_var
    api_url = 'https://backend69.up.railway.app/get/projects'

    # Make the GET request
    response = requests.get(api_url)
    if response.status_code == 200:
        global_var = response.json()  # Assuming the response contains JSON data
    else:
        print("Error: Failed to retrieve data from the API")
        
    # storing the backend data to global_var   
    global_var = global_var.get('message')


# Function to check plagiarism
def check_plagiarism(user_data, project_data):
    # Tokenization, Stemming, and Stopword Removal
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # function to preprocess the data
    def preprocess_text(text):
        tokens = word_tokenize(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        filtered_tokens = [token for token in stemmed_tokens if token.lower() not in stop_words]
        return ' '.join([token.lower() for token in filtered_tokens])

    # Preprocess user input
    user_data = preprocess_text(user_data)
    
    # Preprocess project descriptions
    project_descriptions = [preprocess_text(project_data)]
    
    # Calculate cosine similarity
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([user_data] + project_descriptions)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    
    # caonerting the cosine similarity into float
    plagiarism_score=float(cosine_similarities)
    
    return plagiarism_score

# API endpoint to check plagiarism
@app.route('/get', methods=['POST'])
def api_check_plagiarism():
    try:
        
        global global_var
        if global_var is None:
            get_data_from_api()      

        # taking the data from user upload project form
        user_title = request.form.get('title')
        user_short_description = request.form.get('shortdescription')
        user_description = request.form.get('description')

        # creating boolean accept to check if the project is accepted or not 
        accept=True
        
        # max_similarity stores the max value of every itaration to use it if the project is not rejected in the accepted response
        max_similarity=[]
        for item in global_var:
            if 'title' in item:
                title= item['title']
            if "shortdiscription" in item:
                shortdiscription = item["shortdiscription"]
            if 'description' in item:
                discription = item['description']
               
            plagiarism_score=[0,0,0]
            
            # Check plagiarism
            plagiarism_score[0] = check_plagiarism(user_title, title)
            
            plagiarism_score[1] = check_plagiarism(user_short_description, shortdiscription)
            
            plagiarism_score[2] = check_plagiarism(user_description, discription)
            
            plagiarism_score=max(plagiarism_score)
           
            # add the max value in the max_similarity
            max_similarity.append(plagiarism_score)
            
            # Determine if the project should be accepted or rejected (adjust threshold as needed)
            max_similarity_score = plagiarism_score
            
            max_similarity_score=float(max_similarity_score)
            
            if max_similarity_score > 0.7:
                response = [{'status': 'rejected', 'message': 'Plagiarism detected! Project rejected.','percentage':int(max_similarity_score*100)}]
                accept=False
                break
        
        # if the project is not rejected then it will run with the response project accepted
        if accept :
            max_similarity=max(max_similarity)          
            response = [{'status': 'accepted', 'message': 'Project submitted successfully.','percentage':int(max_similarity*100)}]
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=False)  # Enable debugging for development
