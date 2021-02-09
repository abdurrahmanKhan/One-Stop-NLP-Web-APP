import streamlit as st 
import os
from textblob import TextBlob 
import spacy
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import pandas as pd 
from faker import Faker
import base64
import time 
timestr = time.strftime("%Y%m%d-%H%M%S")
import spacy
from spacy import displacy
nlp = spacy.load('en')



def analyze_text(text):
	return nlp(text)



def make_downloadable_df_format(data,format_type="csv"):
        datafile = data.to_csv(index=False)
        b64 = base64.b64encode(datafile.encode()).decode()
        st.markdown("### ** Download File  📩 ** ")
        new_filename = "fake_dataset_{}.{}".format(timestr,format_type)
        href = f'<a href="data:file/{format_type};base64,{b64}" download="{new_filename}">Click Here!</a>'
        st.markdown(href, unsafe_allow_html=True)


def generate_profile(number,random_seed=200):
	fake = Faker()
	Faker.seed(random_seed)
	data = [fake.simple_profile() for i in range(number)]
	df = pd.DataFrame(data)
	return df 

def generate_locale_profile(number,locale,random_seed=200):
	locale_fake = Faker(locale)
	Faker.seed(random_seed)
	data = [locale_fake.simple_profile() for i in range(number)]
	df = pd.DataFrame(data)
	return df 


def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


@st.cache
def text_analyzer(my_text):
	nlp = spacy.load('en')
	docx = nlp(my_text)
	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
	return allData


@st.cache
def entity_analyzer(my_text):
	nlp = spacy.load('en')
	docx = nlp(my_text)
	tokens = [ token.text for token in docx]
	entities = [(entity.text,entity.label_)for entity in docx.ents]
	allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
	return allData


def main():

	st.title("Your one stop NLP App")

	expander1 = st.beta_expander("Tokenization & Named Entity Recognition")
	with expander1:
                message = st.text_area("Your Text Below")
                col1, col2 = st.beta_columns(2)
                with col1:
                        col1.header("Tokenize Your Text")
                        if st.button("Show Entities"):
                                nlp_result = text_analyzer(message)
                                st.json(nlp_result)
                with col2:
                        col2.header("NER")
                        if st.button("Analyze"):
                                docx = analyze_text(message)
                                html = displacy.render(docx,style="ent")
                                html = html.replace("\n\n","\n")
                                st.write(HTML_WRAPPER.format(html),unsafe_allow_html=True)


	# Sentiment Analysis
	expander2 = st.beta_expander("View Sentiment")
	with expander2:
                st.subheader("Polarity:  It determines if the text expresses the positive, negative or neutral")
                st.markdown(''' #### value lies in the range of [-1,1]
         1  : positive statement
         0  : neutral statement
        -1  : negative statement.''')
                st.subheader("Subjectivity: It determines if the text is subjective or objective")
                st.markdown(''' #### value lies in the range of [0,1]
          0  : Subjective (Has emotions)
          1  : Objective (Fact)''')
                
                
                message2 = st.text_area("Enter Your Text Below")
                if st.button("Show Sentiment metrices"):
                        blob = TextBlob(message2)
                        result_sentiment = blob.sentiment
                        st.success(result_sentiment)

	# Summarization
	expander3 = st.beta_expander("Summarize your Text")
	with expander3:
                message3 = st.text_area("Add your text below")
                summary_options = st.selectbox("Choose Summarizer",['Sumy','Gensim'])
                if st.button("Summarize"):
                        if summary_options == 'Sumy':
                                st.text("Using Sumy Summarizer ..")
                                summary_result = sumy_summarizer(message3)
                        elif summary_options == 'Gensim':
                                st.text("Using Gensim Summarizer ..")
                                summary_result = summarize(message3)
                        else:
                                st.warning("Using Default Summarizer")
                                st.text("Using Gensim Summarizer ..")
                                summary_result = summarize(message3)
                        st.success(summary_result)

	# Dummy Data Generator
	expander4 = st.beta_expander("Generate Dummy Data")
	with expander4:
                column1, column2, column3 = st.beta_columns(3)
                with column1:
                        number_to_gen = st.number_input("Number",5,5000)
                with column2:
                        localized_providers = ["ar_AA", "ar_EG", "ar_JO", "ar_PS", "ar_SA", "bg_BG", "bs_BA", "cs_CZ", "de", "de_AT", "de_CH", "de_DE", "dk_DK", "el_CY", "el_GR", "en", "en_AU", "en_CA", "en_GB", "en_IE", "en_IN", "en_NZ", "en_PH", "en_TH", "en_US", "es", "es_CA", "es_ES", "es_MX", "et_EE", "fa_IR", "fi_FI", "fil_PH", "fr_CA", "fr_CH", "fr_FR", "fr_QC", "he_IL", "hi_IN", "hr_HR", "hu_HU", "hy_AM", "id_ID", "it_CH", "it_IT", "ja_JP", "ka_GE", "ko_KR", "la", "lb_LU", "lt_LT", "lv_LV", "mt_MT", "ne_NP", "nl_BE", "nl_NL", "no_NO", "or_IN", "pl_PL", "pt_BR", "pt_PT", "ro_RO", "ru_RU", "sk_SK", "sl_SI", "sv_SE", "ta_IN", "th", "th_TH", "tl_PH", "tr_TR", "tw_GH", "uk_UA", "zh_CN", "zh_TW"]
                        locale = st.multiselect("Select Locale",localized_providers,default="en_IN")
                with column3:
                        profile_options_list = ['username', 'name', 'sex' , 'address', 'mail' , 'birthdate''job', 'company', 'ssn', 'residence', 'current_location', 'blood_group', 'website']
                        profile_fields = st.multiselect("Fields",profile_options_list,default=['username','mail'])
                   
                custom_fake = Faker(locale)
                data = [custom_fake.profile(fields=profile_fields) for i in range(number_to_gen)]
                df = pd.DataFrame(data)

                st.dataframe(df)

                if st.button("Download"):
                        make_downloadable_df_format(df)
st.sidebar.header("About Author")
st.sidebar.subheader("Abdurrahman")
st.sidebar.text("mailtoabdurrahman24x7@gmail.com")
	

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

if __name__ == '__main__':
	main()
