
import spacy

def process_sentence(sent):
    nlp = spacy.load('en')
    doc = nlp(sent)
    for t in doc:
        print(t)


if __name__ == "__main__":
    process_sentence("Errr, yeah, I haven't been to the loo for quite a time, quite a while, maybe 3 or 4 days")
