"""Predefined name library for swarm subagents.

Each subagent gets a random human first name from this pool at swarm start.
Names are assigned without replacement via :func:`assign_names`.  The
orchestrator tracks names used in prior turns so that no name is reused
across turns within a single session.
"""

from __future__ import annotations

import random

# ~1 075 diverse human first names, sorted alphabetically.
AGENT_NAMES: list[str] = [
    # --- A ---
    "Ada", "Adaeze", "Adair", "Adam", "Adeline",
    "Adira", "Adrian", "Adriana", "Agatha", "Agnes",
    "Ahmad", "Aiden", "Aina", "Ainsley", "Akiko",
    "Akira", "Alana", "Albert", "Alejandro", "Alessia",
    "Alex", "Alfred", "Ali", "Alice", "Alina",
    "Aliyah", "Alma", "Alonso", "Althea", "Alvaro",
    "Amalia", "Amanda", "Amara", "Amber", "Amelia",
    "Amir", "Amira", "Amos", "Amy", "Ana",
    "Anastasia", "Anders", "Andre", "Andrea", "Andrei",
    "Andrew", "Angela", "Anika", "Anisa", "Anjali",
    "Anna", "Antonio", "Anwar", "Anya", "Aoife",
    "April", "Aquila", "Archer", "Arden", "Ariadne",
    "Ariel", "Arjun", "Arlo", "Armand", "Artemis",
    "Arthur", "Arya", "Asha", "Asher", "Ashton",
    "Astrid", "Athena", "Atlas", "Atticus", "Audrey",
    "August", "Aurora", "Autumn", "Ava", "Avery",
    "Axel", "Ayana", "Ayesha", "Ayumi", "Aziza",
    "Azure",
    # --- B ---
    "Bao", "Barbara", "Barrett", "Basil", "Bastian",
    "Bea", "Beatrice", "Beau", "Beckett", "Bella",
    "Benedict", "Benjamin", "Bennett", "Benoit", "Bernadette",
    "Bernard", "Beryl", "Beth", "Bianca", "Bjorn",
    "Blaine", "Blair", "Blake", "Blanca", "Blythe",
    "Bob", "Bodhi", "Boris", "Boyd", "Braden",
    "Bradley", "Brady", "Brandon", "Bree", "Brendan",
    "Brenna", "Brent", "Brett", "Brian", "Bridget",
    "Brigitte", "Brock", "Brooke", "Bruce", "Bruno",
    "Bryan", "Bryce", "Brynn", "Bryson", "Burak",
    # --- C ---
    "Cadence", "Cai", "Caleb", "Calla", "Callum",
    "Calvin", "Cameron", "Camila", "Camille", "Cara",
    "Carlos", "Carmen", "Carolina", "Carson", "Carter",
    "Casey", "Cassandra", "Catalina", "Catherine", "Cecilia",
    "Cedric", "Celeste", "Celia", "Chandra", "Charles",
    "Charlie", "Charlotte", "Chase", "Chen", "Chiara",
    "Chloe", "Chris", "Christian", "Christina", "Christine",
    "Christopher", "Clara", "Clarence", "Clarissa", "Clark",
    "Claude", "Claudia", "Clay", "Clement", "Cleo",
    "Clint", "Clover", "Cody", "Cole", "Colin",
    "Colleen", "Colton", "Connor", "Conrad", "Cora",
    "Cordelia", "Corey", "Corinne", "Cruz", "Crystal",
    "Curtis", "Cyrus",
    # --- D ---
    "Dagny", "Dahlia", "Daisy", "Dakota", "Dale",
    "Dallas", "Dalton", "Damian", "Damon", "Dana",
    "Daniel", "Daniela", "Dante", "Daphne", "Dara",
    "Darcy", "Darian", "Darius", "Darnell", "Darren",
    "David", "Dawson", "Dean", "Deborah", "Declan",
    "Deja", "Delilah", "Demetrius", "Denise", "Dennis",
    "Derek", "Desmond", "Destiny", "Devon", "Diana",
    "Diego", "Dimitri", "Dina", "Dirk", "Dominic",
    "Dominique", "Donald", "Donovan", "Dora", "Dorian",
    "Dorothy", "Douglas", "Drake", "Drew", "Duncan",
    "Dustin", "Dylan",
    # --- E ---
    "Eamon", "Earl", "Easton", "Ebony", "Eden",
    "Edgar", "Edith", "Edmund", "Eduardo", "Edward",
    "Edwin", "Eileen", "Einar", "Elaine", "Eleanor",
    "Elena", "Eli", "Eliana", "Elias", "Elijah",
    "Elina", "Elise", "Elizabeth", "Ella", "Ellen",
    "Elliot", "Eloise", "Elsa", "Elton", "Ember",
    "Emeka", "Emery", "Emi", "Emil", "Emilia",
    "Emily", "Emma", "Emmanuel", "Emmett", "Enid",
    "Enrique", "Enya", "Eric", "Erica", "Erin",
    "Ernest", "Esme", "Esperanza", "Estela", "Esther",
    "Ethan", "Etienne", "Eugene", "Eva", "Evan",
    "Eve", "Evelyn", "Everett", "Ezekiel", "Ezra",
    # --- F ---
    "Fabian", "Faith", "Fallon", "Fang", "Fareed",
    "Faris", "Fatima", "Faye", "Federico", "Felicity",
    "Felix", "Fern", "Fernando", "Finn", "Fiona",
    "Fletcher", "Flora", "Florence", "Floyd", "Flynn",
    "Ford", "Forrest", "Francesca", "Francis", "Francisco",
    "Frank", "Franklin", "Fraser", "Freya", "Frida",
    "Fritz", "Fumiko",
    # --- G ---
    "Gabriel", "Gabriella", "Gael", "Gaia", "Garrett",
    "Gary", "Gavin", "Gemma", "Gene", "Genevieve",
    "George", "Georgia", "Gerald", "Gia", "Gideon",
    "Gina", "Giovanni", "Giselle", "Glen", "Gloria",
    "Godfrey", "Gordon", "Grace", "Graham", "Grant",
    "Gregory", "Greta", "Griffin", "Guadalupe", "Guillermo",
    "Gunnar", "Gustav", "Gwen", "Gwyneth",
    # --- H ---
    "Hadley", "Hana", "Hank", "Hannah", "Hans",
    "Harlan", "Harley", "Harmony", "Harold", "Harper",
    "Harrison", "Harry", "Harvey", "Hassan", "Haven",
    "Hayden", "Hazel", "Heath", "Heather", "Hector",
    "Heidi", "Helen", "Helena", "Hendrix", "Henrik",
    "Henry", "Herbert", "Herman", "Hiro", "Holly",
    "Homer", "Hope", "Horatio", "Howard", "Hudson",
    "Hugh", "Hugo", "Hunter", "Hyacinth",
    # --- I ---
    "Ian", "Ida", "Idris", "Ignacio", "Igor",
    "Iker", "Ilana", "Imani", "Imogen", "India",
    "Indira", "Ines", "Ingrid", "Iona", "Ira",
    "Irene", "Iris", "Irving", "Isaac", "Isabel",
    "Isabella", "Isaiah", "Isla", "Ismail", "Ivan",
    "Ivy",
    # --- J ---
    "Jabari", "Jace", "Jack", "Jackson", "Jacob",
    "Jacqueline", "Jade", "Jaden", "Jaime", "Jake",
    "James", "Jamie", "Jana", "Jane", "Janelle",
    "Janet", "Jared", "Jasmine", "Jason", "Jasper",
    "Javier", "Jay", "Jayden", "Jean", "Jemma",
    "Jennifer", "Jensen", "Jeremiah", "Jeremy", "Jerome",
    "Jesse", "Jessica", "Jethro", "Jett", "Jia",
    "Jillian", "Joanna", "Joaquin", "Jocelyn", "Joel",
    "Johan", "Johanna", "John", "Jolene", "Jon",
    "Jonas", "Jonathan", "Jordan", "Jorge", "Jose",
    "Josephine", "Joshua", "Joy", "Joyce", "Juan",
    "Juanita", "Jude", "Julia", "Julian", "Juliana",
    "Julie", "Juliet", "June", "Juniper", "Justice",
    "Justin",
    # --- K ---
    "Kade", "Kai", "Kaia", "Kaitlyn", "Kalani",
    "Kaleb", "Kamala", "Kamari", "Kamila", "Kane",
    "Kara", "Karen", "Karl", "Karson", "Katarina",
    "Kate", "Katherine", "Katrina", "Kay", "Kayla",
    "Keanu", "Keegan", "Keith", "Kelly", "Kelsey",
    "Kendall", "Kendra", "Kenji", "Kennedy", "Kenneth",
    "Kent", "Kenya", "Kenzo", "Keri", "Kerry",
    "Kevin", "Khalid", "Khalil", "Kian", "Kiera",
    "Kim", "Kimani", "Kimberly", "Kingston", "Kira",
    "Kirk", "Kirsten", "Kit", "Koa", "Kofi",
    "Koji", "Kora", "Krishna", "Kristen", "Kunal",
    "Kurt", "Kyla", "Kylie", "Kyra",
    # --- L ---
    "Lacey", "Laila", "Lana", "Lance", "Lane",
    "Lara", "Larissa", "Lars", "Laura", "Lauren",
    "Laurence", "Lawrence", "Layla", "Leah", "Leandro",
    "Lee", "Leia", "Leif", "Lena", "Leo",
    "Leon", "Leonard", "Leonardo", "Leona", "Leopold",
    "Leroy", "Leslie", "Levi", "Lewis", "Liam",
    "Lila", "Liliana", "Lillian", "Lily", "Lincoln",
    "Linda", "Lionel", "Lisa", "Liv", "Livia",
    "Logan", "Lola", "London", "Lora", "Lorenzo",
    "Lorna", "Louisa", "Louise", "Luca", "Lucia",
    "Lucian", "Lucille", "Lucinda", "Lucy", "Luis",
    "Lukas", "Luke", "Luna", "Luther", "Lydia",
    "Lyle", "Lynette", "Lynn", "Lyra",
    # --- M ---
    "Mabel", "Mack", "Macy", "Maddox", "Madeline",
    "Madison", "Mae", "Maeve", "Magdalena", "Magnus",
    "Maia", "Makoto", "Malachi", "Malcolm", "Malik",
    "Mallory", "Mara", "Marc", "Marcel", "Marcela",
    "Marco", "Marcus", "Margaret", "Margot", "Maria",
    "Mariam", "Mariana", "Marie", "Marina", "Mario",
    "Marisol", "Mark", "Marlene", "Marley", "Marlon",
    "Marshall", "Martha", "Martin", "Marvin", "Mason",
    "Mateo", "Matilda", "Matthew", "Matteo", "Maura",
    "Maurice", "Mavis", "Max", "Maxine", "Maxwell",
    "Maya", "McKenna", "Megan", "Mei", "Melanie",
    "Melissa", "Melody", "Mercedes", "Mercy", "Meredith",
    "Mia", "Michael", "Michelle", "Miguel", "Mika",
    "Miles", "Milo", "Mina", "Minerva", "Miranda",
    "Miriam", "Mitchell", "Moira", "Molly", "Monica",
    "Monroe", "Morgan", "Moses", "Murat", "Murray",
    "Mustafa", "Myles", "Myra",
    # --- N ---
    "Nadia", "Nadine", "Naomi", "Nash", "Nasir",
    "Natalia", "Natalie", "Natasha", "Nathan", "Nathaniel",
    "Naveen", "Neil", "Nell", "Nelson", "Nessa",
    "Neva", "Nevin", "Newton", "Nicholas", "Nico",
    "Nicole", "Nigel", "Nikita", "Nikolai", "Nila",
    "Nina", "Noa", "Noah", "Noel", "Noelle",
    "Nolan", "Nora", "Noreen", "Norman", "Nova",
    "Nuri", "Nyla",
    # --- O ---
    "Oakley", "Oberon", "Octavia", "Odessa", "Odette",
    "Odin", "Olga", "Olive", "Oliver", "Olivia",
    "Omar", "Oona", "Ophelia", "Ora", "Oren",
    "Orion", "Orlando", "Oscar", "Otis", "Otto",
    "Owen",
    # --- P ---
    "Pablo", "Paige", "Paloma", "Pamela", "Parker",
    "Pascal", "Patricia", "Patrick", "Paul", "Paula",
    "Paulina", "Pavel", "Paxton", "Pearl", "Pedro",
    "Penelope", "Pepper", "Percival", "Percy", "Perla",
    "Perry", "Peter", "Petra", "Peyton", "Philip",
    "Phoebe", "Phoenix", "Pierce", "Piper", "Portia",
    "Preston", "Priscilla", "Priya",
    # --- Q ---
    "Qadir", "Qiana", "Quentin", "Quincy", "Quinn",
    "Quinlan",
    # --- R ---
    "Rachel", "Rae", "Rafael", "Raiden", "Raina",
    "Raj", "Ralph", "Ramona", "Randall", "Raphael",
    "Rashid", "Raul", "Raven", "Ray", "Raymond",
    "Reagan", "Rebecca", "Reed", "Reese", "Regina",
    "Reid", "Remy", "Rena", "Renata", "Rene",
    "Rex", "Reyna", "Rhea", "Rhiannon", "Rhys",
    "Ricardo", "Richard", "Riley", "Rio", "Rita",
    "River", "Robert", "Robin", "Rocco", "Roderick",
    "Rodrigo", "Roger", "Roland", "Roman", "Romeo",
    "Ronan", "Rosa", "Rosalie", "Rosalind", "Rose",
    "Rosemary", "Rowan", "Roxana", "Roy", "Ruby",
    "Rufus", "Rupert", "Russell", "Ruth", "Ryan",
    "Ryder",
    # --- S ---
    "Sabine", "Sabrina", "Sadie", "Sage", "Said",
    "Sakura", "Salim", "Sally", "Salvador", "Sam",
    "Samantha", "Samir", "Samira", "Samuel", "Sana",
    "Sandra", "Sanjay", "Santiago", "Santos", "Sara",
    "Sarah", "Sasha", "Saul", "Savannah", "Sawyer",
    "Scarlett", "Scott", "Sean", "Sebastian", "Selena",
    "Serena", "Sergio", "Seth", "Shane", "Shannon",
    "Sharon", "Shawn", "Shea", "Shelby", "Shepherd",
    "Shiloh", "Shirley", "Sibyl", "Sidney", "Sierra",
    "Silas", "Simeon", "Simon", "Simone", "Skyler",
    "Sloane", "Sofia", "Solomon", "Sonia", "Sophia",
    "Soren", "Spencer", "Stella", "Stephanie", "Sterling",
    "Steven", "Stone", "Stuart", "Suki", "Sullivan",
    "Summer", "Sung", "Susan", "Sutton", "Sven",
    "Sylvia", "Sylvie",
    # --- T ---
    "Tabitha", "Talia", "Talon", "Tamara", "Tamsin",
    "Tane", "Tanya", "Tara", "Tate", "Tatiana",
    "Taylor", "Teagan", "Ted", "Teresa", "Terrence",
    "Terry", "Tessa", "Thaddeus", "Thalia", "Thea",
    "Theodore", "Theresa", "Thomas", "Thor", "Tia",
    "Tiana", "Tierney", "Timothy", "Tobias", "Todd",
    "Tomas", "Tony", "Torin", "Travis", "Trevor",
    "Tristan", "Troy", "Tucker", "Tyler",
    # --- U ---
    "Ulrich", "Uma", "Umberto", "Una", "Uri",
    "Uriel", "Ursula", "Usha",
    # --- V ---
    "Valentina", "Valeria", "Valerie", "Vance", "Vanessa",
    "Vaughn", "Vera", "Veronica", "Vesper", "Victor",
    "Victoria", "Viggo", "Vince", "Vincent", "Viola",
    "Violet", "Virginia", "Vivian", "Vladimir",
    # --- W ---
    "Wade", "Walker", "Wallace", "Walter", "Wanda",
    "Ward", "Warren", "Wayne", "Wendy", "Wesley",
    "Weston", "Whitney", "Wilbur", "Wilder", "Willa",
    "William", "Willow", "Wilson", "Winifred", "Winston",
    "Winter", "Wolfgang", "Wyatt", "Wynn",
    # --- X ---
    "Xander", "Xanthe", "Xavier", "Xena", "Ximena",
    "Xiu", "Xochitl",
    # --- Y ---
    "Yael", "Yamato", "Yara", "Yareli", "Yasmin",
    "Yolanda", "Yoshi", "Yousef", "Yuan", "Yuki",
    "Yumiko", "Yuri", "Yusuf", "Yvette", "Yvonne",
    # --- Z ---
    "Zachariah", "Zachary", "Zadie", "Zahara", "Zainab",
    "Zane", "Zara", "Zaria", "Zelda", "Zena",
    "Zephyr", "Zia", "Zinnia", "Zion", "Zoe",
    "Zora", "Zuri",
]


def assign_names(n: int, *, exclude: set[str] | None = None) -> list[str]:
    """Return *n* unique random names from the pool.

    Parameters
    ----------
    n:
        Number of names to assign.
    exclude:
        Names already used in previous turns.  These are removed from the
        candidate pool so that subagents in a new turn never share a name
        with subagents from an earlier turn.

    If *n* exceeds the available pool size the result is capped.
    """
    pool = AGENT_NAMES if not exclude else [name for name in AGENT_NAMES if name not in exclude]
    return random.sample(pool, min(n, len(pool)))
