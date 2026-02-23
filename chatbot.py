import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Predefined intents
intents = {
    "hello": "Hi there! How can I help you?",
    "how are you": "I'm just a bot, but I'm doing great!",
    "what is ai": "Artificial Intelligence is the simulation of human intelligence in machines.",
    "bye": "Goodbye! Have a nice day!"
}

questions = list(intents.keys())
responses = list(intents.values())

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    index = np.argmax(similarity)

    if similarity[0][index] < 0.3:
        return "Sorry, I don't understand that."
    else:
        return responses[index]

print("AI Chatbot (type 'exit' to quit)")

while True:
    user_input = input("You: ").lower()
    if user_input == "exit":
        break
    print("Bot:", chatbot_response(user_input))
