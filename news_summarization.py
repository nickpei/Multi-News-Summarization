import requests
import bs4 as bs
import urllib.request
import json
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk import pos_tag
import string
import time
from math import log10
from google import search
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import *

# define stop_words for stemming
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def googleSearch(query):
    """ send query to google and use request library to get the response href
    query: input key phrase, string
    type query: string
    rtype: list of 
    """
    return [url for url in search( query, stop=20)]

def generate_top_k(web_info, top_k, document_number):
    """generate top K importance word for the document
       type web_info: bs4.BeautifulSoup
       type top_k: list
       type document_number: int
       rtype: dictionary
    """
    title = web_info.title.string
    word_list = word_tokenize(title)
    for word in word_list:
        if word in string.punctuation:
            pass
        elif word not in stop_words:
            word = ps.stem(word)
            if word not in top_k.keys():
                top_k[word] = 1
            else:
                top_k[word] += 1 
    return top_k

def modify_top_k(top_k):
    """extend the top_k words, calculate each word TF
    score and update the dictionary
    type top_k: dict
    rtype: dict
    """
    total = sum(top_k.values())
    extented_dict = {}
    for word in top_k.keys():
        top_k[word] = top_k[word]/total
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                try:
                    l = ps.stem(l)
                    if l not in top_k.keys():
                        extented_dict[l] = top_k[word]
                except:
                    pass
    top_k_updated = top_k.copy()
    top_k_updated.update(extented_dict)
    return top_k_updated

def generate_first_10(information,document_number, first_10_sentence):
    """
    generate the first 10 sentence for each link for later analysis
    type informaiton: list
    type document_number: int
    type first_10_sentence: list
    rtype: list
    """
    for p in information[:11]:
        sent_array = sent_tokenize(p.text)
        if len(sent_array) != 0:
            sent_array.append(document_number)
            first_10_sentence.append(sent_array)
    return first_10_sentence

def generate_middle(information, document_number, middle_sentence):
    """
    generate the first 10 sentence for each link for later analysis
    type informaiton: list
    type document_number: int
    type middle_sentence: list
    rtype: list
    """
    for p in information[11:]:
        sent_array = sent_tokenize(p.text)
        if len(sent_array) != 0:
            text = word_tokenize(sent_array[0])
            tagged = pos_tag(text)
            for word in tagged:
                if word[1] == "MD":
                    sent_array.append(document_number)
                    middle_sentence.append(sent_array)
                    break
    return middle_sentence

def generate_scores(sentence_list, top_k):
    """generate score for each sentence in the list
    type sentence_list: list
    type top_k: dict
    rtype: list"""
    for sentence in sentence_list:
        text = word_tokenize(sentence[0])
        score = 0
        for word in text:
            word = ps.stem(word)
            if word in top_k.keys():
                score = score + top_k[word]
        sentence.insert(0, score)
    return sentence_list

def main():
    """run the main application
    """
    query = input("What do you want to search?\n")
    links_array = googleSearch(query)
    web_crawl_fail = 0
    top_k = {}
    first_10_sentence = []
    middle_sentence = []
    for document_number,url in enumerate(links_array):
        print(document_number)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            link = urllib.request.urlopen(req).read()
            web_info = bs.BeautifulSoup(link,'lxml')
            try:
                top_k = generate_top_k(web_info, top_k, document_number)
            except:
                print("Title Extraction Error")
            information = web_info.find_all("p")
            try:
                first_10_sentence = generate_first_10(information, document_number, first_10_sentence)
            except:
                print("first_10_sentence error")
            try:
                middle_sentence = generate_middle(information, document_number, middle_sentence)
            except:
                print("middel_sentence error")               
        except:
            print("Web Crawling Error")
            web_crawl_fail = web_crawl_fail + 1
            
    #calculate the scores, and sort them
    top_k = modify_top_k(top_k)
    first_10_sentence = generate_scores(first_10_sentence,top_k)
    middle_sentence = generate_scores(middle_sentence, top_k)

    first_10_sentence = sorted(first_10_sentence, key=itemgetter(0), reverse = True)
    first_10_sentence = first_10_sentence[0:31]
    middle_sentence = sorted(middle_sentence, key=itemgetter(0), reverse = True)
    impact_list = [middle_sentence[n][1] for n in range(25)]

    #cosine similairty
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(impact_list)
    similarity_score = cosine_similarity(tfidf_matrix, tfidf_matrix)
    #print(similarity_score[0:1])
    similarity_score = similarity_score.tolist()
    
    #combination the word length limitation and maximum marginal relevence
    #to generate the summary
    print("the summary is: ")
    max_word = 800
    summary = []
    summary.append(first_10_sentence[0][1])
    summary.append(impact_list[0])
    print("{}\n {}".format(summary[0],summary[1]))
    max_word = max_word - len(summary)
    n = 0
    while max_word >= 0:
        for similarity in similarity_score[n][n:]:
            if similarity < 0.1:
                #if the similarity is smaller than 0.1 then select the sentences
                n = similarity_score[n].index(similarity)
                break
        sentence = impact_list[n]
        print(sentence)
        max_word = max_word - len(sentence)
        summary.append(sentence)
        
    # save the information to files
    with open("word.txt","w") as f:
        f.truncate()
        json.dump(top_k, f)
    with open("first_10_sentence.txt","w") as f:
        f.truncate()
        json.dump(first_10_sentence, f)
    with open("middle_sentence.txt","w") as f:
        f.truncate()
        json.dump(middle_sentence, f)
    with open("impact_list","w") as f:
        f.truncate()
        json.dump(impact_list, f)
        
main()
