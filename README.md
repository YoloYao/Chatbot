# Chatbot
Generative conversation robot based on NLP.

You can select one of the following user names to enter during login:

    Alice, Beffy, Dylan, Lucy, Mike

After entering the project directory in the terminal, you can choose between two modes to start the Chatbot:

## Command line mode start command:

    python3 main.py

## User interface boot mode:

#### Terminal 1 Background service:

    python3 service/app.py

#### Terminal 2 Front desk Service:

    cd web
    python3 -m http.server 8000

After the foreground service is started, you can enter the local address in the browser to open the interface:
    
#### http://localhost:8000/
