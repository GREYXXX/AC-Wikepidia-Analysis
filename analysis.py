import torch
import torch.nn.functional as F
import re
from bs4 import BeautifulSoup
from pandas import merge
import requests
import spacy
from sentence_transformers import SentenceTransformer, util

print("loading spaCy and SBERT model....")
nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('all-MiniLM-L6-v2')
print("loading finished")

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

class Solver:

    def __init__(self, names):
        """Init variables"""
        self.base  = "https://en.wikipedia.org/wiki/"
        self.names = names
        self.texts = dict()
        
        for name in self.names:
            self.texts[name] = ''

    def fill_text_helper(self, link):
        """
        Helper functions to extract and preprocess the text crawl from wiki page
        """
        try: 
            page = requests.get(link)
        except Exception as e:
            print(f"request failure with the link {link} and error {e}")
            return

        soup = BeautifulSoup(page.content, 'html.parser')
        text = ''
        for paragraph in soup.find_all('p'):
            text += paragraph.get_text() 

        text = text.replace('\n', '')
        text =  re.sub('\[\d+\]', '', text.strip())
        return text
    
    def fill_text(self):
        """
        Filling the processed texts to the dict that has the key named by members
        """
        for name in self.names:
            self.texts[name] = self.fill_text_helper(self.base + name)
    
    def get_adjective_found_helper(self, name):
        """
        extract the adjs from texts with 2 forms:
        1. merged noun chunk docs 
        2. unmerged noun chunk docs
        merge noun chunk can merge the words like "Western Australia" into one 
        single word, or it will be tokenized as "western" and "australia"
        """
        docs = nlp(self.texts[name]) 
        adjs_no_merge = [token.text for token in docs if token.pos_ == 'ADJ']
        with docs.retokenize() as retokenizer:  
            for chunk in docs.noun_chunks:
                retokenizer.merge(chunk) 
        adjs_merged = [token.text for token in docs if token.pos_ == 'ADJ']
        return (adjs_no_merge, adjs_merged)

    def print_adjective_found(self):
        """Prints the adjs along each member's name"""
        for name in self.names:
            adjs_no_merge = self.get_adjective_found_helper(name)[0]
            adjs_merged   = self.get_adjective_found_helper(name)[1]
            print(f"Adjs found for {name} are:")
            print("-------------------------------------------------------")
            print(f"no merge chunk: {adjs_no_merge}")
            print("\n")
            print(f"merged chunk: {adjs_merged}")
            print("-------------------------------------------------------")
            print("\n\n")
  
    
    def cut_sentence(self):
        """Cut sentence"""
        merged_corpus = '.'.join([self.texts[name] for name in self.names])
        self.sents = split_into_sentences(merged_corpus)

    def encode_sentence(self):
        """
        Because we need to print the cosine similarity with orginial member's 
        wiki page, so I made a hash table to look up the value (name) of 
        key (each sentence).
        """
        self.merged_corpus = dict()
        for name in self.names:
            sents = split_into_sentences(self.texts[name])
            for sent in sents:
                self.merged_corpus[sent] = name

    def calc_cosine(self):
        """
        1. enable pytorch cuda to speed up the calc of embedding and cosine 
        similarity.
        2. print the sentence that originally belongs to member's wiki page with
        cosine similarity >= 0.8. In this case, I just print the cosine 
        similarity with member's name because that is easier to visulize. And if
        we know the member's name of the sentence then we know the orginal 
        wiki page.
        """   
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
          print("Using CUDA")

        #Compute embeddings
        print("embedding start")
        embeddings = model.encode(self.sents, convert_to_tensor=True).to(device)
        torch.nn.DataParallel(embeddings)
        print("embedding done")

        #Compute cosine-similarities for each sentence with each other sentence
        print("calc start")
        cosine_scores = util.cos_sim(embeddings, embeddings).to(device)
        torch.nn.DataParallel(cosine_scores)
        print("calc done")

        print("pairs start")
        pairs = []
        for i in range(len(cosine_scores)-1):
            for j in range(i+1, len(cosine_scores)):
                pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
        print("pairs done")

        for pair in pairs[:]:
            try:
                i, j = pair['index']
                if pair['score'] >= 0.8 and (self.sents[i] != self.sents[j]):
                    print("{} \t\t {} \t\t Score: {:.4f}".format(self.get_name_sentence(self.sents[i]), self.get_name_sentence(self.sents[j]), pair['score']))
                else:
                    continue
            except Exception as e:
                print(f"exception with sentence {self.sents[i]} and \t{self.sents[i]} with \t{e}")

    def get_text(self):
        """for the test purpose"""
        return self.texts
 
    def get_name_sentence(self, sent):
        """for test purpose"""
        return self.merged_corpus[sent]

    def get_cut_sentence(self):
        """for test purpose"""
        return self.sents


if __name__ == "__main__":

    # All the members of Australia cabinet from wiki page
    names = ["Scott_Morrison", "Barnaby_Joyce", "Josh_Frydenberg", "Simon_Birmingham", "David_Littleproud", "Peter_Dutton", 
            "Marise_Payne", "Greg_Hunt", "Michaelia_Cash", "Dan_Tehan", "Paul_Fletcher_(politician)", "Karen_Andrews", "Angus_Taylor_(politician)",
            "Ken_Wyatt", "Anne_Ruston", "Linda_Reynolds", "Sussan_Ley", "Stuart_Robert", "Alan_Tudge", "Melissa_Price_(politician)",
            "Bridget_McKenzie", "Andrew_Gee", "Alex_Hawke", "Keith_Pitt"]

    obj = Solver(names)
    obj.fill_text()
    obj.cut_sentence()
    obj.encode_sentence()
    obj.print_adjective_found() 
    obj.calc_cosine()
