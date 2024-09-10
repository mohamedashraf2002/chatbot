
import nltk
import spacy
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
     
nlp=spacy.load('en_core_web_sm')
faqs=[
    {"question": "What are the working days, hours and holidays??","answer": "We are open from 9 AM to 6 PM, From Monday to Friday @ sunday is off."},
    {"question": "How can I contact customer support?","answer": "You can contact our customer support at email support@services.com OR 1956."},
    {"question": "There is a problem with the Internet?","answer": "okay,l will active the line now ."},
    {"question": "Can I change my line?","answer": "Unfortunately, you can change your line same the number."},
    {"question": "l want call the manger?", "answer": "okay l will transfer you to manger."},
    {"question": "Are they any offers now?", "answer": "Yes, There are offers up to 50%."},
    {"question": "What offers do you have?", "answer": "30% discount on internet and other services."},
]
     
questions=[faq['question'] for faq in faqs]
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(questions)
def advanced_preprocess(text):
  doc=nlp(text)
  tokens =[token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
  return ' '.join(tokens)

def chatbot_response_advanced(user_query):
  user_query_processed=advanced_preprocess(user_query)
  user_query_vec=vectorizer.transform([user_query_processed])
  similarities=cosine_similarity(user_query_vec,X)
  max_sim_index=similarities.argmax()
  return faqs[max_sim_index]['answer']
     

def chat_advanced():
    print("Welcome to the FAQ chatbot. Type 'exit' to end our conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = chatbot_response_advanced(user_input)
        print("Bot:", response)

