import streamlit as st
import os
import requests
import atexit 

save_path = r"C:\Users\vivek_pankaj\Desktop\DAP\dap_app_v1.0\src\files"

# Define the different views
def home_view():
    st.title("Home")
    st.write("Welcome to the home page!")

def delete_from_index():
    url = "http://localhost:8000/delete"
    response = requests.get(url)
    return response.json()

def add_to_index(file_path):
    url = "http://localhost:8000/add"
    response = requests.post(url, json={"file_path": file_path})
    return response.json()

def upload_view():
    st.title("Upload Files")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    save_button = st.button("Save")

    #show all the files in save_path
    files_already_uploaded = os.listdir(save_path)
    if len(files_already_uploaded) > 0:
        st.write("Files already uploaded:")
        for file in files_already_uploaded:
            st.write(file)
    
    if save_button and uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(save_path, file.name), "wb") as f:
                f.write(file.getbuffer())
        st.write("Files saved successfully!")
        #add a loading bar
        for file in uploaded_files:
            add_to_index(os.path.join(save_path, file.name))
        st.write("Files added to the index successfully!")

def termination():
    print("Deleting the files from storage")
    files_already_uploaded = os.listdir(save_path)
    for file in files_already_uploaded:
        os.remove(os.path.join(save_path, file))

atexit.register(termination)

def chatbot_view():
    st.title("Chatbot")
    st.write("Welcome to the chatbot page!")

    # Text input for user query
    user_query = st.text_input("Enter your query:")

    # Button to submit the query
    if st.button("Submit"):
        # Placeholder for chatbot response
        response = get_chatbot_response(user_query)
        st.write("Chatbot response:", response)

# Placeholder function for chatbot response
def get_chatbot_response(query):
    chatbot_ip = "http://localhost:8000/qa"
    try:
        response = requests.post(chatbot_ip, json={"question": query})
        return response.json()["answer"]
    except Exception as e:
        return {"error": str(e)}

# Create a sidebar for navigation
st.sidebar.title("Navigation")
view = st.sidebar.radio("Go to", ["Home", "Upload Files", "Chatbot"])

# Display the selected view
if view == "Home":
    home_view()
elif view == "Upload Files":
    upload_view()
elif view == "Chatbot":
    chatbot_view()