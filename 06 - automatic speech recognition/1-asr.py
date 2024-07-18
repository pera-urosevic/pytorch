import matplotlib.pyplot as plt
from model import bundle, predict
from decoder import Decoder

if __name__ == '__main__':
    prediction = predict('~data/portal.ogg')

    plt.imshow(prediction.cpu().T)
    plt.title("Classification")
    plt.xlabel("Frame")
    plt.ylabel("Class")
    plt.show()

    decoder = Decoder(labels=bundle.get_labels())
    text = decoder.decode(prediction)
    print(f'speech recognition:\n{text}')
