
from itertools import chain, combinations
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from itertools import zip_longest
from IPython.display import HTML
import nltk
from nltk.stem import PorterStemmer
from nltk.data import find
nltk.download('word2vec_sample')

# Load Google's pre-trained Word2Vec model - most common ~44k words
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = KeyedVectors.load_word2vec_format(
    word2vec_sample, binary=False)

ps = PorterStemmer()

best_guess = ""
best_words = []
best_sim = 0

red_words = ["lava", "cave", "man"]
black_words = ["rail"]

# Filter words that are not in the vocab of the model
def filter_words(words):
    filtered_words = []
    for word in words:
        try:
            _ = model[word]
            filtered_words.append(word)
        except KeyError:
            print(word, "not in vocab")
            continue
    return filtered_words

filtered_red = filter_words(red_words)
# filtered_blue = filter_words(blue_words)
filtered_black = filter_words(black_words)
# filtered_beige = filter_words(beige_words)

# Return powerset of words from start len to end len
def powerset(iterable,start,end):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(start,end + 1)))

# Search from 1 or 2 to 4 word combinations
start = 1 if len(red_words) == 1 else 2
possible_answers = powerset(filtered_red, start, 4)

for possible_answer in possible_answers:
    guesses = model.most_similar(positive=list(possible_answer), negative=filtered_black)

    # Use stemming to filter out words that are from the same root word
    final_guess = []
    print(guesses)
    for guess in guesses:
        final_guess = guess
        valid_guess = True
        for word in possible_answer:
            if(ps.stem(word) == ps.stem(guess[0]) or word.lower() in guess[0].lower() or guess[0].lower() in word.lower()):
                valid_guess = False
                break
        if valid_guess:
            break

    if final_guess[1] > best_sim:
        print(guesses)
        best_guess = final_guess[0]
        best_words = possible_answer
        best_sim = final_guess[1]

print(best_guess, best_words, best_sim)