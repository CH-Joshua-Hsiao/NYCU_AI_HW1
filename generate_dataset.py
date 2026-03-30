import csv
import random

def generate_dataset(num_samples=800):
    # To guarantee uniqueness, we will use sets
    small_queries = set()
    large_queries = set()
    
    target_each = num_samples // 2

    # --- SMALL LLM EXPANDED COMPONENTS ---
    countries = ["France", "Germany", "Japan", "Brazil", "Canada", "Australia", "India", "Egypt", "South Korea", "Italy", "Spain", "Mexico", "Argentina", "Nigeria", "Kenya", "South Africa", "China", "Vietnam", "Thailand", "Indonesia", "Peru", "Chile", "Colombia", "Sweden", "Norway", "Finland", "Denmark", "Poland", "Ukraine", "Greece"]
    languages = ["French", "Spanish", "German", "Italian", "Portuguese", "Russian", "Japanese", "Korean", "Mandarin", "Arabic", "Hindi", "Bengali", "Turkish", "Vietnamese", "Dutch"]
    phrases = ["hello", "goodbye", "thank you", "please", "yes", "no", "water", "food", "help", "friend", "good morning", "how are you", "I love you", "where is the bathroom", "excuse me", "I am sorry", "what is your name", "nice to meet you", "see you later", "how much does this cost"]
    people = ["Albert Einstein", "Isaac Newton", "Marie Curie", "Nikola Tesla", "Leonardo da Vinci", "George Washington", "Abraham Lincoln", "Mahatma Gandhi", "Nelson Mandela", "Martin Luther King Jr.", "Winston Churchill", "Charles Darwin", "Galileo Galilei", "William Shakespeare", "Aristotle", "Plato", "Socrates", "Alexander the Great", "Julius Caesar", "Cleopatra"]
    terms = ["photosynthesis", "gravity", "democracy", "inflation", "osmosis", "mitosis", "entropy", "algorithm", "bandwidth", "metabolism", "capitalism", "socialism", "communism", "fascism", "anarchism", "existentialism", "nihilism", "stoicism", "utilitarianism", "pragmatism"]
    cities = ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Moscow", "Beijing", "Dubai", "Mumbai", "Los Angeles", "Chicago", "Toronto", "Seoul", "Sao Paulo", "Buenos Aires", "Mexico City", "Cairo", "Lagos", "Istanbul"]
    books = ["Romeo and Juliet", "To Kill a Mockingbird", "1984", "Pride and Prejudice", "The Great Gatsby", "Moby Dick", "Hamlet", "The Odyssey", "War and Peace", "The Catcher in the Rye", "Don Quixote", "The Divine Comedy", "The Iliad", "The Brothers Karamazov", "Crime and Punishment", "Anna Karenina", "Jane Eyre", "Wuthering Heights", "Great Expectations", "A Tale of Two Cities"]
    chemicals = ["water", "carbon dioxide", "sulfuric acid", "salt", "glucose", "methane", "ammonia", "hydrochloric acid", "sodium hydroxide", "ethanol", "methanol", "propane", "butane", "octane", "benzene", "toluene", "phenol", "acetone", "formaldehyde", "acetic acid"]
    landmarks = ["Mount Everest", "the Eiffel Tower", "the Burj Khalifa", "the Statue of Liberty", "the Great Pyramid of Giza", "the Empire State Building", "the Leaning Tower of Pisa", "the Washington Monument", "the Space Needle", "the CN Tower", "the Golden Gate Bridge", "the Colosseum", "the Taj Mahal", "the Great Wall of China", "Machu Picchu", "Stonehenge", "the Acropolis", "the Kremlin", "the Louvre", "the Sydney Opera House"]

    # --- LARGE LLM EXPANDED COMPONENTS ---
    libraries = ["BeautifulSoup", "Pandas", "Flask", "PyTorch", "Selenium", "TensorFlow", "Scikit-Learn", "Django", "FastAPI", "Keras", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly", "Requests", "Urllib", "Pytest", "Unittest", "SQLAlchemy"]
    tasks = ["scrape a dynamic webpage", "merge two large datasets", "build a REST API", "train a CNN", "automate form filling", "perform sentiment analysis", "cluster customer data", "deploy a machine learning model", "create an interactive dashboard", "fetch data from a GraphQL endpoint", "implement user authentication", "manage database migrations", "optimize matrix multiplication", "visualize geographic data", "test an asynchronous function"]
    constraints = ["pagination and rate limits", "missing values and mixed data types", "user authentication and database migrations", "CUDA out of memory errors and vanishing gradients", "captchas and random class names", "imbalanced classes and overfitting", "high dimensionality and sparse matrices", "docker containerization and CI/CD pipelines", "real-time updates and low latency", "complex nested JSON structures", "secure password hashing and JWT tokens", "schema evolution and backward compatibility", "hardware acceleration and memory bottlenecks", "custom map projections and large shapefiles", "mocking external APIs and flaky tests"]
    
    comparisons_A = ["capitalism", "relational databases", "quantum computing", "agile methodology", "stoicism", "Python", "React", "AWS", "TCP", "REST"]
    comparisons_B = ["socialism", "NoSQL databases", "classical computing", "waterfall methodology", "nihilism", "JavaScript", "Angular", "Azure", "UDP", "GraphQL"]
    contexts = ["income inequality", "horizontal scaling", "cryptography", "project risk management", "finding meaning in life", "data science applications", "single page applications", "cloud computing costs", "streaming video", "API design"]
    
    essays = ["the 2008 financial crisis", "the Protestant Reformation", "climate change", "the fall of the Roman Empire", "the rise of artificial intelligence", "the Industrial Revolution", "the French Revolution", "the American Civil War", "World War II", "the Cold War"]
    essay_factors = ["the housing bubble, deregulation, and predatory lending", "corruption in the Catholic Church, the printing press, and political fragmentation", "greenhouse gas emissions, deforestation, and industrialization", "economic instability, military overstretch, and political corruption", "advances in computing power, the availability of large datasets, and algorithmic innovations", "technological innovation, capital accumulation, and labor migration", "social inequality, economic hardship, and enlightenment ideas", "states' rights, slavery, and economic divergence", "fascism, appeasement, and unresolved WWI grievances", "ideological conflict, nuclear deterrence, and proxy wars"]
    
    debug_langs = ["React", "C++", "Java", "Python", "SQL", "Rust", "Go", "TypeScript", "Ruby", "PHP"]
    debug_errors = ["Maximum update depth exceeded", "Segmentation fault", "NullPointerException", "IndentationError", "Syntax error", "Borrow checker error", "Panic: index out of bounds", "Type 'X' is not assignable to type 'Y'", "NoMethodError", "Parse error"]
    debug_situations = ["updating state in a useEffect loop", "dereferencing a null pointer", "calling a method on an uninitialized object", "mixing spaces and tabs", "using a reserved keyword as a column name", "mutating a borrowed value", "accessing a slice incorrectly", "passing the wrong props to a component", "calling a misspelled function", "missing a semicolon"]

    # Hard Small LLM prefixes
    hard_small_prefixes = [
        "Hey there! I am currently working on a massive, complex system architecture project that spans multiple continents. During my deep research into various networking protocols and data pipelines, I suddenly couldn't remember something very basic. Could you please tell me ",
        "I'm debugging a huge React codebase with nested Redux states, complex useEffect hooks, and hundreds of custom components. While doing that, I got a message from my colleague. Can you just ",
        "Consider a scenario where the macroeconomic impacts of a 2008-style financial crisis cause rapid market deflation. In completely unrelated news, I need to know: ",
        "I am writing a detailed essay about the structural factors causing climate change, including greenhouse gas emissions, deforestation, and industrialization. As a fun aside for my introduction, ",
        "Act as a senior software architect designing a scalable distributed file system using NoSQL and Kubernetes. Oh wait, actually, just ",
        "Compare and contrast the philosophical implications of stoicism versus nihilism when confronting the absurdity of modern life. Actually, skip all that and just ",
        "Please provide a Python script that uses BeautifulSoup to scrape Wikipedia for articles on machine learning, extracts the first paragraph of each, and saves the text to a Postgres database. Just kidding, I only need to know: ",
    ]

    # Hard Large LLM generic specific queries (add a random UUID or number to make them strictly unique if drawn multiple times)
    hard_large_base = [
        "Prove computationally that P != NP given constraints of system {}.",
        "Write a robust Regex for matching valid IPv6 addresses complying with RFC {}.",
        "Explain quantum entanglement mathematically in the context of tensor product space {}.",
        "Build a Redux slice for a shopping cart containing {}.",
        "Code a generic binary search tree in Rust prioritizing memory safety struct {}.",
        "Analyze O(n log n) vs O(n^2) dynamically for array size {}.",
        "Implement a custom React hook for debouncing an input field with delay {}ms.",
        "Provide the Navier-Stokes equations for incompressible flow modeling scenario {}.",
        "Configure an Nginx reverse proxy with SSL terminating at port {}."
    ]

    while len(small_queries) < target_each:
        rand_val = random.random()
        if rand_val < 0.15: # Hard small LLM
            prefix = random.choice(hard_small_prefixes)
            # Pick a random simple question to append
            base_q = random.choice([
                f"what is the capital of {random.choice(countries)}?",
                f"translate '{random.choice(phrases)}' to {random.choice(languages)}.",
                f"when was {random.choice(people)} born?",
                f"how tall is {random.choice(landmarks)}?"
            ])
            q = prefix + base_q
        else:
            q = random.choice([
                f"What is the capital of {random.choice(countries)}?",
                f"How do you say '{random.choice(phrases)}' in {random.choice(languages)}?",
                f"When was {random.choice(people)} born?",
                f"Define {random.choice(terms)}.",
                f"What time is it in {random.choice(cities)}?",
                f"Who wrote {random.choice(books)}?",
                f"Translate {random.choice(phrases)} to {random.choice(languages)}.",
                f"What is the formula for {random.choice(chemicals)}?",
                f"How tall is {random.choice(landmarks)}?",
                f"Convert {random.randint(1,100)} Celsius to Fahrenheit.",
                f"What is the square root of {random.randint(2, 500)}?"
            ])
        small_queries.add(q)

    while len(large_queries) < target_each:
        if random.random() < 0.15: # Hard large LLM
            base_q = random.choice(hard_large_base)
            q = base_q.format(random.randint(1000, 9999))
        else:
            q = random.choice([
                f"I need a Python script using {random.choice(libraries)} to {random.choice(tasks)}. Ensure it handles {random.choice(constraints)}.",
                f"Compare and contrast {random.choice(comparisons_A)} with {random.choice(comparisons_B)} in the context of {random.choice(contexts)}.",
                f"Write a detailed essay about the structural factors causing {random.choice(essays)}. Discuss at least {random.choice(essay_factors)}.",
                f"Can you debug this {random.choice(debug_langs)} code? It's throwing a '{random.choice(debug_errors)}' error when {random.choice(debug_situations)}.",
                f"Develop a comprehensive business plan for a {random.choice(['fintech', 'healthtech', 'edtech', 'greentech', 'foodtech'])} startup, including analyzing {random.choice(['market fit', 'burn rate', 'VC funding', 'competitors'])}.",
                f"Explain the mathematical proof for {random.choice(['the Pythagorean theorem', 'the irrationality of the square root of 2', 'Fermat\'s Last Theorem', 'the fundamental theorem of calculus'])} and how it applies to modern systems.",
                f"Act as a senior software architect. Design a system for {random.choice(['a global e-commerce platform', 'a real-time messaging app', 'a ride-sharing service'])} that prioritizes {random.choice(['horizontal scaling', 'zero-downtime deployments', 'data redundancy'])}.",
                f"Analyze the historical significance of {random.choice(essays)} and its impact on global economics.",
                f"Summarize the recent research developments in {random.choice(['natural language processing', 'renewable energy', 'gene editing', 'materials science', 'astrophysics'])} and predict how it will affect the next decade."
            ])
        large_queries.add(q)

    # Combine and save
    data = []
    for q in small_queries:
        data.append({"query": q, "label": "small-LLM", "label_id": 0})
    for q in large_queries:
        data.append({"query": q, "label": "large-LLM", "label_id": 1})

    random.shuffle(data)
    
    with open('dataset.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query', 'label', 'label_id'])
        writer.writeheader()
        writer.writerows(data)
        
    print(f"Successfully generated {len(data)} perfectly UNIQUE queries and saved to dataset.csv")

if __name__ == '__main__':
    generate_dataset()
