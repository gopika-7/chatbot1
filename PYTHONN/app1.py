# app1.py

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 09:13:11 2025

@author: Gopika
"""

import streamlit as st
import random
import joblib
from datetime import datetime
from textblob import TextBlob
import time
from PIL import Image
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize training data if not already present
if 'X_data' not in globals():
    X_data = []  # This should be your original training data
if 'y_data' not in globals():
    y_data = []  # This should be your original labels

# Function to get personalized greeting
def get_greeting():
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good morning! How can I assist you today?"
    elif 12 <= current_hour < 17:
        return "Good afternoon! How can I assist you today?"
    elif 17 <= current_hour < 21:
        return "Good evening! How can I assist you today?"
    else:
        return "Hello! How can I assist you tonight?"

# Function to load an image (chatbot avatar)
def load_image(image_path):
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img = img.resize((50, 50))  # Resize for consistency
        return img
    else:
        return None

# Function for typing animation effect
def typing_animation(text, delay=0.1):
    for char in text:
        st.write(char, end="", flush=True)
        time.sleep(delay)
    st.write()  # Newline at the end

# Function to add emojis based on intent
def add_emoji(response, tag):
    emojis = {
        "greeting": "😊",
        "age": "🎂",
        "name": "💬",
        "farewell": "👋",
        "programming": "💻",
        "math": "➗",
        "tech": "🔧",
        "help": "🆘"
    }
    return response + " " + emojis.get(tag, "")

# Function for sentiment analysis
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

# Function for random fun facts
def get_fun_fact():
    facts = [
        "Did you know? Honey never spoils!",
        "Did you know? A shrimp's heart is in its head!",
        "Here's a fun fact: The Eiffel Tower can grow taller during the summer due to the expansion of the iron."
    ]
    return random.choice(facts)

# Function for random jokes
def get_joke():
    jokes = [
        "Why don’t skeletons fight each other? They don’t have the guts!",
        "I told my wife she was drawing her eyebrows too high. She looked surprised!",
        "Why don’t oysters donate to charity? Because they’re shellfish!"
    ]
    return random.choice(jokes)

# Load trained model and vectorizer
clf = joblib.load('classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to update the classifier with new data
def update_classifier():
    new_intents = add_additional_intents()
    for intent in new_intents:
        for pattern in intent["patterns"]:
            X_data.append(pattern)
            y_data.append(intent["tag"])

    # Re-train the model with the updated data
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_data)
    
    # Train your classifier (use the same model you're using currently)
    clf = MultinomialNB()
    clf.fit(X_train, y_data)
    
    # Save the updated model and vectorizer
    joblib.dump(clf, 'updated_classifier_model.pkl')
    joblib.dump(vectorizer, 'updated_vectorizer.pkl')

# Adding new tags and responses
def add_additional_intents():
    additional_intents = [
        {"tag": "programming", "patterns": ["What is Python?", "Tell me about Java", "Explain C++", "How do I use Python?", "What is a variable in programming?"], "responses": ["I can help with programming! What would you like to know about Python, Java, or C++?"]},
        {"tag": "math", "patterns": ["What is 2+2?", "Solve 3*3", "What is 100 divided by 5?", "Can you solve an equation for me?"], "responses": ["I can solve math problems! Ask away."]},
        {"tag": "tech", "patterns": ["Tell me about Artificial Intelligence", "What is Cloud Computing?", "Explain blockchain", "What is 5G?", "What is IoT?"], "responses": ["Technology is always advancing! Let me know if you want information on AI, blockchain, IoT, or any tech topic."]},
        {"tag": "help", "patterns": ["Can you help me?", "I need help", "Assist me please", "Help with my question", "Can you guide me?"], "responses": ["I’m here to help! Please ask your question, and I’ll do my best to assist."]},
    ]
    return additional_intents

# Function to display bot's response
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    response = ""
    
    # Adding responses for different tags
    if tag == "greeting":
        response = "Hello! How can I assist you today?"
    elif tag == "age":
        response = "I am ageless! I exist to help you."
    elif tag == "name":
        response = "I am your friendly chatbot, always here to assist you!"
    elif tag == "farewell":
        response = "Goodbye! Have a great day!"
    elif tag == "programming":
        response = "Are you looking for programming help? I can assist with Python, Java, C++, and more!"
    elif tag == "math":
        response = "Sure, I can help with basic math! Ask me a math question."
    elif tag == "tech":
        response = "Tech is evolving rapidly! What tech topic are you interested in? AI, Cloud, or something else?"
    elif tag == "help":
        response = "Sure, I’m here to help. Please ask me anything, and I’ll try my best to assist!"
    
    # Add emojis based on intent
    response = add_emoji(response, tag)
    return response

# Streamlit App
def main():
    # Sidebar for navigation
    menu = ["Home", "About", "Conversation History"]
    choice = st.sidebar.selectbox("Select an option", menu)
    
    if choice == "Home":
        st.title("Chatbot")
        st.write(get_greeting())  # Display personalized greeting
        
        user_name = st.text_input("What’s your name?")
        if user_name:
            st.write(f"Hello, {user_name}! Let's chat.")
        
        counter = 0  # For multiple conversations
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display user input and bot response in a vertical layout
        for message in st.session_state.chat_history:
            # Displaying each message in a vertical flow
            st.write(f"{message}")  # Each message will appear on a new line
        
        user_input = st.text_input("You:", key=f"user_input_{counter}")
        if user_input:
            # Add user input to chat history
            st.session_state.chat_history.append(f"You: {user_input}")
            
            # Determine sentiment
            sentiment = get_sentiment(user_input)
            if sentiment == "positive":
                response = "I’m happy to hear that! 😊"
            elif sentiment == "negative":
                response = "I’m sorry to hear that. Let me know how I can help. 😔"
            else:
                response = chatbot(user_input)
            
            # Add chatbot response to chat history
            st.session_state.chat_history.append(f"Chatbot: {response}")
            
            # Display response directly (without typing animation)
            st.write(response)
            
            # Clear chat on button click
            if st.button('Clear Chat'):
                st.session_state.chat_history = []
                st.rerun()  # Replaces st.experimental_rerun() with st.rerun()
            
            # Add functionality for fun facts and jokes
            if "fact" in user_input.lower():
                response = get_fun_fact()
                st.session_state.chat_history.append(f"Chatbot: {response}")
                st.write(response)
            elif "joke" in user_input.lower():
                response = get_joke()
                st.session_state.chat_history.append(f"Chatbot: {response}")
                st.write(response)
            
            # End conversation when the user says "goodbye"
            if "goodbye" in user_input.lower():
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()
    
    elif choice == "About":
        st.title("About")
        st.write("""The chatbot is a simple program designed to interact with users and provide helpful responses. It can answer questions, provide fun facts, tell jokes, and even analyze the sentiment of the user's message (whether it's positive, negative, or neutral). The chatbot can greet users, assist with queries related to various topics like programming and math, and respond based on the user's input. It can also remember the conversation history, making it easy to see what was discussed earlier. The chatbot aims to make conversations engaging and informative in a friendly and helpful way.""")
    
    elif choice == "Conversation History":
        st.title("Conversation History")
        if 'chat_history' in st.session_state:
            for message in st.session_state.chat_history:
                st.write(message)
        else:
            st.write("No conversation history yet.")

# Run the app
if __name__ == '__main__':
    main()
