import os
import pandas as pd
import pprint
import csv
import fitz
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('punkt')



#pdf to text 
def extract_text(pdf):
    doc = fitz.open(pdf)
    file_name= os.path.basename(pdf)

    doc_text = "\n\n".join([page.get_text() for page in doc])  

    #remove enters (\n) but keeping \n\n
    doc_text = doc_text.replace('\n\n', '<<PAGE>>')
    doc_text = doc_text.replace('\n' , ' ')
    doc_text = doc_text.replace('<<PAGE>>', '\n\n') 
    doc_text = f"{file_name}\n\n{doc_text}"


    #removing speciall characters
    #doc_text = doc_text.replace('\n','').replace('\t', '').replace('\ufeff', '').replace('\x08', '')
    #doc_text = doc_text.replace('\t', '').replace('\ufeff', '').replace('\x08', '')

     #further cleaning
    #phrases = ["AI", "system", "including", "eli", "http", "article", "annex", "chapter", "oj", "relevant", "point", "based", "following", "thereafter", "iii", "i", "considered", "eu", "en", "november",
     #          "regard", "ensure", "referred", "information", "due", "additional", "accordance", "amending"  ]
    #for phrase in phrases:
     #   doc_text = doc_text.replace(phrase, "")

    doc_text_filename = os.path.splitext(pdf)[0] + ".txt"
    with open(doc_text_filename, "w", encoding="utf-8") as text_file:
        text_file.write(doc_text)  
    return doc_text



#extract metadata from pdf
def extract_metadata(pdf):
    doc = fitz.open(pdf)
    doc_metadata = doc.metadata
    return doc_metadata


def remove_phrases(text, phrases):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in " ".join(phrases).split()]
    return " ".join(tokens)


#split a document into sentences
def split_to_sentence(text_path):
   with open(text_path, "r", encoding="utf-8") as file:
    text = file.read()
    #tokenize the text file into sentences
    sentences = sent_tokenize(text) 
    return sentences

#split a document into pages
def split_into_pages(text_path):
    with open(text_path, "r", encoding="utf-8") as file:
        text = file.read()
        pages = text.split("\n\n")  
    return [p.strip() for p in pages if p.strip()]   

# Save to CSV
def save_csv(sentences, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Header"]) #header
        for sentence in sentences:
            if sentence.strip(): #to avoid \n\n
                writer.writerow([sentence])  
    return csv_file        


#extract sentences from csv - don't need it 
def extract_sentences(csv_path): 
    sentences =[]
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                sentences.append(row[0])
    return sentences              

def preprocess_text(sentences): 
    processed_sentences = []
   
    #stop_words = set(stopwords.words('english'))
   
    for sentence in sentences:
        #  3. Remove URLs
        sentence = re.sub(r'https?://\S+|www\.\S+', '', sentence)
        #1.tokenising
        tokens = word_tokenize(sentence)
        #2.lowercasing and removing non-alphabetic tokens
        tokens = [token.lower() for token in tokens if token.isalpha()] #Does this remove words with hyphens?? 
        #3.lemmatization
       # lemmatizer = WordNetLemmatizer()
       # tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # 5. Remove stop words
       # tokens = [token for token in tokens if token not in stop_words]

      

        # 4. Remove numbers, but not words that contain numbers.
        tokens = [token for token in tokens if not token.isnumeric()]
        # 5. Remove words that are only one character.
        tokens = [token for token in tokens if len(token) > 1]
        processed_sentences.append(' '.join(tokens))    
    
    return processed_sentences





extract_text('doc.pdf')
filename = os.path.splitext('doc.pdf')[0]
sentences = split_to_sentence(filename+'.txt')
save_csv(sentences , filename+"_sen.csv")

save_csv(preprocess_text(sentences), filename+"_sen_clean.csv") #pre-processed sentences

pages = split_into_pages(filename+'.txt')
save_csv(pages, filename+"_page.csv")
save_csv(preprocess_text(pages), filename+"_page_clean.csv") #pre-processed pages




