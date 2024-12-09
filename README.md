# Chatbot
Generative conversation robot based on NLP

There are two modes of operation: 

**command line question answering mode** and **user interface**

#### Command line mode start command:

    python3 main.py

User interface boot mode:

#### Terminal 1 Background service:

    python3 service/app.py

#### Terminal 2 Front desk Service:

    python -m http.server 8000

After the foreground service is started, you can enter the local address in the browser to open the interface:
    
#### http://localhost:8000/