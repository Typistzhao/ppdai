# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 00:23:33 2015

@author: nausheenfatma
"""

import re
import operator

class Bigrams:
    def __init__(self):
        self.f1=None #sample file to read
        self.bigrams_dict={}                  #Dictionary to hold bigrams and its count
        self.dict_plot={}                      #x=key=rank,y=value=frequency
        self.ranked_list=[] #bigrams
        self.bigrams_feature_vector={}
        self.bigrams_ranked_list_of_words=[]
        self.last_token=""      
        self.outputpath=None
        self.inputpath=None
        
    def set_input_path(self,inputpath):
        self.inputpath=inputpath
        
    def set_output_path(self,outputpath):
        self.outputpath=outputpath        
        
        
    def add_bigrams_to_list(self,token,main_word):  #unigrams_dict is of the form {token:[frequency,case satisfied,last word from which token generated]}
        if self.last_token != "":
            bigram="("+self.last_token+","+token+")"
            if bigram not in self.bigrams_dict :
                self.bigrams_dict[bigram]=[1,bigram]
            else :
                count=self.bigrams_dict[bigram][0]+1
                self.bigrams_dict[bigram]=[count,bigram]
        self.last_token=main_word 

    def find_grams(self,token):
       # print token
        self.add_bigrams_to_list(token,token)
        
    def sort_descending(self,bigrams):
        return sorted(bigrams.items(), key=operator.itemgetter(1), reverse=True)


    def find_x_y_for_plotting(self,ranked_list,grams_dict,filepath):
        dict_plot={}
        f1=open(filepath, "w")
        for i in range(len(ranked_list)):
            word=ranked_list[i][0]                #fetching the word string from ranked list
            dict_plot[i+1]=grams_dict[word][0] #finding the frequency of word from bigrams_dict
            f1.write(word+"\t\t:\t"+str(grams_dict[word][0])+"\n")
        
        f1.close()
        return dict_plot


    def find_bigrams(self):
        self.f1=open(self.inputpath,"r")
        i=1
        for line in self.f1:
            self.last_token=""
            if i%1000==0:
                print(i)
            i=i+1
            tokens=line.rstrip().split()
            for token in tokens :
                self.find_grams(token)

        self.ranked_list=self.sort_descending(self.bigrams_dict)  #key=word,value=frequency
        #self.ranked_list_2=self.sort_descending(self.bigrams_dict)  #key=word,value=frequency
        
        bigrams_output_path=self.outputpath
        self.find_x_y_for_plotting(self.ranked_list,self.bigrams_dict,bigrams_output_path)

def main():
    u=Bigrams()
    u.set_input_path("char_doc.txt")
    u.set_output_path("Bigrams_char.txt")
    u.find_bigrams()
        
if __name__ == '__main__':
    main()
