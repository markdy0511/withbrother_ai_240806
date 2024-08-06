import requests
from bs4 import BeautifulSoup
import asyncio
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.chains.llm import LLMChain
from langchain.schema import BaseOutputParser, output_parser
from langchain_community.document_loaders import YoutubeLoader
from langchain.storage import LocalFileStore
from langchain.document_loaders import TextLoader
from langchain.schema import StrOutputParser
from langchain.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
import insert_logo

asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
strict_llm = ChatOpenAI(
    model="gpt-4-1106-preview",
    temperature=0,
) #요약 같은 정확한 업무를 해야 할 때

flexible_llm = ChatOpenAI(
    model="gpt-4-1106-preview",
    temperature=0.2,
) #요약 같은 정확한 업무를 해야 할 때

mellting_llm = ChatOpenAI(
    model="gpt-4-1106-preview",
    temperature=0.5,
) #스크립트 작성 같은 창의력을 요하는 업무를 해야 할 때

bs_transformer = BeautifulSoupTransformer()

def load_naver_news(url): #네이버 뉴스에서 필요한 정보 추출
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser') #네이버 뉴스 html
    # 네이버 뉴스에서 제목, 작성 기관, 작성일, 작성자, 기사 내용을 추출    
    try:
        title = soup.select("#title_area > span")[0].get_text()
        ref = soup.select("#ct > div.media_end_head.go_trans > div.media_end_head_top._LAZY_LOADING_WRAP > a > img.media_end_head_top_logo_img.light_type._LAZY_LOADING._LAZY_LOADING_INIT_HIDE")[0].get("title")
        date = soup.select("#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span")[0].get("data-date-time")
        maker = soup.select("#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_journalist > a > em")[0].get_text()
        context = soup.select("#dic_area")[0].get_text()

        source = {
            "title" : title,
            "ref": ref,
            "date": date,
            "maker": maker,
            "context": context
        }
    except:
        loader = AsyncChromiumLoader([url])
        docs = loader.load()
        docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["article"])
        source = {
        "title" : None,
        "ref": None,
        "date": None,
        "maker": None,
        "context": docs_transformed[0].page_content
    }

    return source

def load_youtube_transcript(url): #유튜브 링크에서 필요한 정보 추출
    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=True,
        language=["en", "ko"],
        translation="ko")
    source = loader.load() #list 형태인 듯 [0:~~~, metadata={'source','title','description','view_count','thumbnail_url','publish_date','length','author'}]

    return source

def source_input_form():
    with st.form("Source_Info_form"):
        reading_level = st.selectbox("What do you want to set level of reading?", ('Easy', 'Normal'))
        article_type = st.selectbox("What type do you want?", ('Journal Press Style', 'Conversational Style'))
        length_summary = st.selectbox("How long do you want the summary to be?", ('Short', 'Normal'))
        num_outline = st.selectbox("How many outlines do you have?", ('1', '2', '3', '4', '5'))
        length_article = st.selectbox("How long do you want a paragraph to be?", ('3 Sentences', '4 Sentences'))
        Intro_Conclu_switch = st.selectbox("Do you want to include Intro and Conclusion?", ('Yes', 'No'))
        opinion = st.text_input("Optional! Do you have added information or opinion of article?")
        #description = st.text_input("What do you want to tell to reader?")
        #num_outline = st.text_input("How many outlines do you have?")
        submitted = st.form_submit_button("Set Writing Material")
    if submitted:
        return reading_level, article_type, length_summary, num_outline, length_article, Intro_Conclu_switch, opinion, submitted
    else:
        return None, None, None, None, None, None, None, None

class ListOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("list", "")
        return text.split("|")
    
list_output_parser = ListOutputParser()


st.set_page_config(
    page_title="Script Maker",
    page_icon="📜",
)


insert_logo.add_logo("withbrother_logo.png")


st.markdown(
    """
    # Script Maker
            
    Use Script Assistant for your magazine work.
            
    Start by writing the URL of the website or/with what you want to write on the sidebar.

    When you choose NAVER or YOUTUBE, you are going to be take summary of the source.

    You cannot change order of outline, yet.
    
"""
)


with st.sidebar: #원하는 소스를 만드는 곳
    submitted = None
    writing_material = {}
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "NAVER NEWS",
            "YOUTUBE VEDIO",
        ),
    )

    if choice == "NAVER NEWS":
        url = st.text_input(
        "Write down a URL and Press Enter",
        placeholder="https://n.news.naver.com/article/~~~",
        )
        if url:
            with st.spinner('Loading Naver News Website...'):
                reading_level, article_type, length_summary, num_outline, length_article, Intro_Conclu_switch, opinion, submitted = source_input_form()
                if submitted:
                    source = load_naver_news(url) # dictionary
                    writing_material = source
                    writing_material["reading_level"] = reading_level
                    writing_material["article_type"] = article_type
                    writing_material["length_summary"] = length_summary
                    writing_material["num_outline"] = num_outline
                    writing_material["length_article"] = length_article
                    writing_material["Intro_Conclu_switch"] = Intro_Conclu_switch
                    writing_material["opinion"] = opinion



    elif choice == "YOUTUBE VEDIO":
        url = st.text_input(
        "Write down a URL and Press Enter",
        placeholder="https://www.youtube.com/~~~",
        )
        if url:
            with st.spinner('Loading Youtube Website...'):
                reading_level, article_type, length_summary, num_outline, length_article, Intro_Conclu_switch, opinion, submitted = source_input_form()
                if submitted:
                    source = load_youtube_transcript(url) # list [0:~~~,metadata={'source','title','description','view_count','thumbnail_url','publish_date','length','author'}]
                    writing_material = {
                    "title" : source[0].metadata['title'],
                    "ref": None,
                    "date": source[0].metadata['publish_date'],
                    "maker": source[0].metadata['author'],
                    "context": source[0].page_content
                    }
                    writing_material["reading_level"] = reading_level
                    writing_material["article_type"] = article_type
                    writing_material["length_summary"] = length_summary
                    writing_material["num_outline"] = num_outline
                    writing_material["length_article"] = length_article
                    writing_material["Intro_Conclu_switch"] = Intro_Conclu_switch
                    writing_material["opinion"] = opinion

    else:
        st.write("Something Wrong, You choose anything. Check it!")

if submitted: #writing_material: #원하는 소스를 묶은 프롬프트용 
    #print(writing_material)
    first_summary_prompt = ChatPromptTemplate.from_template(
        """
        Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:                
    """
    )

    first_summary_chain = first_summary_prompt | strict_llm | StrOutputParser()

    summary = first_summary_chain.invoke(
        {"text": writing_material["context"]},
    )
    #print(summary)

    short_refine_prompt = ChatPromptTemplate.from_template(
        """
        Your job is to produce a final summary.
        We have provided an existing summary up to a certain point: {existing_summary}
        We have the opportunity to refine the existing summary (only if needed) with some more context below.
        The length of summary depends on {length_summary}.
        If {length_summary} is Short, you have to make summary within 100 words and 3 sentences.
        A sentence means containing one content and ending with a point.
        ------------
        {context}
        ------------
        Given the new context, refine the original summary.
        You have to follow the rules of length.

        Print the final summary In KOREAN.
        You must print only the final context.
        A sentence means containing one content and ending with a point.
        """
    )

    normal_refine_prompt = ChatPromptTemplate.from_template(
        """
        Your job is to produce a final summary.
        We have provided an existing summary up to a certain point: {existing_summary}
        We have the opportunity to refine the existing summary (only if needed) with some more context below.
        The length of summary depends on {length_summary}.
        If {length_summary} is not Short, you have to make summary within 200 words in KOREAN and 5 sentences.
        A sentence means containing one content and ending with a point.
        ------------
        {context}
        ------------
        Given the new context, refine the original summary.
        You have to follow the rules of length.

        Print the final summary In KOREAN.
        You must print only the final context.
        Don't use the translation tone.
        A sentence means containing one content and ending with a point.
        """
    )

    if writing_material["length_summary"] == "Short":
        refine_chain = short_refine_prompt | flexible_llm | StrOutputParser()
    else:
        refine_chain = normal_refine_prompt | flexible_llm | StrOutputParser()

    with st.status("Summarizing...") as status:
        refine_summary = refine_chain.invoke(
                {
                    "existing_summary": summary,
                    "context": writing_material["context"],
                    "length_summary" : writing_material["length_summary"]
                }
            )
    print(refine_summary)
    #result_summary = refine_summary.replace(". ",".<br>")

    sentences = refine_summary.split('. ')
    bullet_list = "<ul>" + "".join(f"<li>{sentence}.</li>" if not sentence.endswith('.') else f"<li>{sentence}</li>" for sentence in sentences if sentence) + "</ul>"
    print(bullet_list)
    st.markdown(bullet_list, unsafe_allow_html=True)

    outline_prompt = ChatPromptTemplate.from_template(
        """
        Your job is to make magazine outline as many as {num}.
        Genre of the magazine is "lifestyle" for ordinary people.
        The outline means a topic of paragraph.
        We have provided an format example of outline below.
        ------------
        1. Describe and define main topic and keywords
        2. Mention about conclusion or result of the article
        3. Mention about opposite opinion or detail explanation of the article
        4. Conclude with future outlook or predictions
        ------------
        You should modify the format of outline to suit context below and the number of outline, {num}. 
        We have infomation to generate magazine outline with some more context below.
        Also, We have writer's opinion together.
        You have to generate outline with given opinion.
        if you didn't take opinion, you should use only context.
        ------------
        {context}
        {opinion}
        ------------
        Given the context, generate the magazine outline as many as {num}.

        Print outline In KOREAN

        if you take num is 6, print final 6 outline example like:
        1. print First Session title : Describe it.
        2. print Second Session title : Describe it.
        3. print Third Session title : Describe it.
        4. print Fourth Session title : Describe it.
        5. print Fifth Session title : Describe it.
        6. print Sixth Session title : Describe it.

        ''((num is 3, you create only 3 outlines))''

        YOU HAVE TO GENERATE ONLY {num} outlines!!!

        """
    )

    outline_chain = outline_prompt | flexible_llm

    outline = outline_chain.invoke(
                {
                    "num": str(writing_material["num_outline"]),
                    "context": writing_material["context"],
                    "opinion":writing_material["opinion"]
                }
            )
    print(outline)
    formatting_outline_prompt = ChatPromptTemplate.from_template(
    """
        You are a powerful formatting algorithm.
        
        You format outline sets into LIST format.
        
        Example Input:

        1. First Session
        2. Second Session
        3. Third Session
        4. Fourth Session
        5. Fifth Session

        Example Output:
        
        ```list
        First Session | Second Session | Third Session | Fourth Session | Fifth Session
        ```
        Your turn!

        CORE MESSEGE: {context}

    """,
    )

    list_outline_chain = formatting_outline_prompt | flexible_llm | list_output_parser

    with st.status("Generating outline...") as status:
        list_outline = list_outline_chain.invoke(
                {
                    "context": outline,
                }
            )
    #st.write(list_outline)

    for i, response in enumerate(list_outline):
        with st.container():
            st.text_area(f"아웃라인 {i+1}",value=response, height=100)


    script_prompt = ChatPromptTemplate.from_template(
            """
            Write a magazine article of the following outline:
            "{text}"
            1 outline must have {length_outline} sentences.
            A sentence means containing one content and ending with a point.
            The final article must consist of an introduction, body, and conclusion according to the outline.
            give me the final magazine article.           
        """
        )

    script_chain = script_prompt | mellting_llm | StrOutputParser()
    with st.status("Generating script...") as status:
        script = script_chain.invoke(
            {"text": outline,
                "length_outline":writing_material["length_article"]},
        )

    easy_level = """You must generate article for students in elementary school.
                    Write a magazine article of the following outline:"""
    normal_level = """Write a magazine article of the following outline:"""
    press_style = """The final article must consist of an introduction, body, and conclusion according to the outline.
            give me the final magazine article."""
    conversation_style ="""The final article must consist of an introduction, body, and conclusion according to the outline.
            give me the final magazine article. As well as, you must change session title to a question format.
            For example, you take original session title, 'Implementation of semiconductor subsidies in the United States'. You have to change like, 'Semiconductor subsidies? What is it?'  """
    yes_in_con = """You have to print with Introduction and Conclusion"""
    no_in_con = """You have to delete Introduction and Conclusion"""
    last_prompt = ""

    if writing_material["reading_level"] == "Easy":
        last_prompt = last_prompt + easy_level
    else:
        last_prompt = last_prompt + normal_level

    if writing_material["article_type"] == "Journal Press Style":
        last_prompt = last_prompt + press_style
    else:
        last_prompt = last_prompt + conversation_style

    if writing_material["Intro_Conclu_switch"] == "No":
        last_prompt = last_prompt + no_in_con
    else:
        last_prompt = last_prompt + yes_in_con

    
    converting_prompt = ChatPromptTemplate.from_template(
            """
            Convert a magazine article of the following script:
            '{script}'
            1 outline must have {length_outline} sentences.
            A sentence means containing one content and ending with a point.
            """ + last_prompt + """
            You have to check and match word and expression from original article: {text}
            Because you can make typo during tranlation.
            You have to print IN KOREAN. 
            Don't use the translation tone.
            And print article using markdown format like below:
            ----------
            # magazine title
            ## session title
            #### article context
            ----------
        """
        )

    convert_chain = converting_prompt | mellting_llm | StrOutputParser()
    
    with st.status("Converting script...") as status:
        result = convert_chain.invoke(
            {"script": script,
                "length_outline":writing_material["length_article"],
                "text": writing_material["context"]}
        )

    st.markdown(result)
