import argparse

import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


def predict(review):
    seq = tokenizer.texts_to_sequences([review])
    seq_pad = pad_sequences(seq, maxlen=1000)
    return model.predict_classes(seq_pad)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict stars of yelp review')
    parser.add_argument('--review')
    args = parser.parse_args()

    model = load_model('keras.model')
    tokenizer = joblib.load('tokenizer.pickle')
    if args.review:
        pred = predict(args.review)
        print('number of stars: {}'.format(pred))

