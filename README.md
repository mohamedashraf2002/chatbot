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


