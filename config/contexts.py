class Contexts:
    BOOK_KEYS = ['buy ticket', 'book ticket', 'reserve ticket']
    DESTINATION_HINT = 'OK! Where would you like to go?'
    TIME_HINT = "Sure! Traveling to [{}]. When do you want to leave?"
    NUM_HINT = "Leaving at [{}]. How many tickets would you like?"
    CONFIRM_HINT = "Booking [{}] ticket(s) to [{}] at [{}]. Are you sure you want to make this reservation? \nEnter [yes] to confirm and [no] to exit."
    SUCCESS_HINT = "Great! Your tickets are booked. This is your book number[{}].Thank you!"
    STOP_HINT = "Ok, if you need to buy tickets later, please ask me."

    DISCOVERY_KEYS = ['service provide',
                      'provide service', 'give service', 'service give']
    DISCOVERY_ANSWER = "Yes, I can offer train ticket booking service. \nPlease enter [I need to book a ticket]  or something similar to activate the corresponding service."
    INTEREST_KEYS = ['what can i do']
    INTEREST_ANSWER = {"piano": "You can try playing the piano for a while to relax.",
                       "violin": "You can try playing the violin for a while to relax.", "basketball": "You can play basketball with your friends for a while. It must be relaxing.", "swim": "You can go swimming. It must be cool.", "film": "You can watch a movie to relax your mind for a while."}
