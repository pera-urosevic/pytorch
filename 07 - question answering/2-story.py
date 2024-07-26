from model import ask, load

if __name__ == "__main__":
    print('## Shape of Things by Ray Bradbury')
    story = load("./~data/shape of things.txt")
    ask(story, "Who is Peter?")
    ask(story, "When we'll have some results?")
    ask(story, "What year is it?")
    ask(story, "Who looks dangerous?")
