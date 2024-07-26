from model import ask, load

if __name__ == "__main__":
    print('## BERT Wikipedia Page')
    bert = load("./~data/bert.txt")
    ask(bert, "BERT is a?")
    ask(bert, "How many modules BERT has?")
    ask(bert, "Is BERT fine-tuned?")
    ask(bert, "Who published BERT?")
    ask(bert, "When was BERT published?")
