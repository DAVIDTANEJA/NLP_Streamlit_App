# NLP App (Tokenization, Named entity recognition, Sentiment analysis, Text summarization)
import streamlit as st

# NLP pkgs
import spacy                                         # tokenization-lemmatization , also can use 'import nltk'
from textblob import TextBlob                        # sentiment analysis

# text summarization
from gensim.summarization import summarize               # gensim
from sumy.parsers.plaintext import PlaintextParser        # sumy
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer  


# tokenization and lemmatization
def text_analyzer(my_text):                
    nlp = spacy.load('en')     # create object of spacy ,   'en_core_web_sm'
    docx = nlp(my_text)                    # use on text
    all_data = [(f"Tokens : {token.text}  and  Lemmas : {token.lemma_}") for token in docx]
    return all_data


# Named entity recognition
@st.cache                                                       # @st.cache : improve performance   
def entity_analyzer(my_text):               
    nlp = spacy.load('en')                      # en_core_web_sm   
    docx = nlp(my_text)
    entites = [(f"Tokens : {entity.text} and  Entity : {entity.label_}") for entity in docx.ents]
    return entites


# sumy - text summarization
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()   # create object
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result



# main app
def main():
    st.title("NLP App using Streamlit")

    # tokenization and lemmatization
    if st.checkbox("Show Tokens and Lemma"):
        message = st.text_area("Tokenize text here :")    # take text
        if st.button("Analyze"):                          # analyze button
            nlp_result = text_analyzer(message)           # using function tokenizer
            st.json(nlp_result)                           # showing result in json format

    # Named Entity recognition
    if st.checkbox("Show Named Entities"):
        message = st.text_area("Extract entities here :")    # take text
        if st.button("Extract"):    
            nlp_result = entity_analyzer(message)     
            st.json(nlp_result)

    # Sentiment analysis
    if st.checkbox("Show Sentiment Analysis"):
        message = st.text_area("Check sentiments here :")    # take text
        if st.button("Analyze"):                             # analyze button
            blob = TextBlob(message)
            result = blob.sentiment
            st.success(result)

    # Text Summarization
    if st.checkbox("Show Text Summarization"):
        message = st.text_area("Summarize the Text")    # take text
        
        summary_options = st.selectbox("Choose summarizer", ("gensim", "sumy"))
        if st.button("Summarize"):
            if summary_options == "gensim":
                st.text("Using Gensim..")
                summary_result = summarize(message)
            elif summary_options == "sumy":
                st.text("Using sumy..")
                summary_result = sumy_summarizer(message)

            else:
                st.warning("Using Default Summarizer")
                st.text("Using Gensim..")
                summary_result = summarize(message)

            st.success(summary_result)          # put outside if-else


    # sidebar
    st.sidebar.subheader("About the App")
    st.sidebar.text("NLP App with Streamlit")
    st.sidebar.info("Cudos to the streamlit Team")


if __name__ == "__main__":
    main()
