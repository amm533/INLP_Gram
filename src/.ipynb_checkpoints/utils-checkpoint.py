

def check_vocabulary(model, words):

    in_vocab_words = []
    for word in words:
        if word not in model:
            print(f" CAUTION: {word} not in model. Removing...")
        else:
            in_vocab_words.append(word)

    print(f"Array after check: {in_vocab_words}")
    print(f"Length of array after check: {len(in_vocab_words)}")

    return in_vocab_words