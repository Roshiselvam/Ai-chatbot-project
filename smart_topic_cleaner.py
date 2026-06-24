# smart_topic_cleaner.py

def preprocess_for_wikipedia(query):
    query = query.strip().lower()

    # Full dictionary of mappings
    mapping = {
        "disadvantages of python": "Criticism of Python (programming language)",
        "advantages of python": "Features of Python (programming language)",
        "duck typing": "Duck typing",
        "chatgpt": "ChatGPT",
        "steve jobs": "Steve Jobs",
        "elon musk": "Elon Musk",
        "india prime minister": "List of Prime Ministers of India",
        "first world war": "World War I",
        "second world war": "World War II",
        "bharat": "India",
        "mk gandhi": "Mahatma Gandhi",
        "nehru": "Jawaharlal Nehru",
        "tamilnadu cm": "Chief Ministers of Tamil Nadu",
        "moon mission": "Chandrayaan programme",
        "mars mission": "Mars Orbiter Mission",
        "dr a p j abdul kalam": "A. P. J. Abdul Kalam",
        "covid": "COVID-19 pandemic",
        "corona": "COVID-19 pandemic",
        "ai": "Artificial intelligence",
        "machine learning": "Machine learning",
        "deep learning": "Deep learning",
        "cr7": "Cristiano Ronaldo",
        "messi": "Lionel Messi",
        "modi": "Narendra Modi",
        "india president": "President of India",
        "budget 2025": "Union budget of India",
        "india map": "Geography of India"
    }

    for key, topic in mapping.items():
        if key in query:
            return topic

    return query
