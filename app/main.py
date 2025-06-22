import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    st.title("📧 Cold Mail Generator")
    
    # Input URL and button for submission
    url_input = st.text_input("Enter a URL:", value="https://jobs.nike.com/job/R-33460")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            # Load and clean the job page content
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)

            # Load portfolio data into ChromaDB
            portfolio.load_portfolio()

            # Extract job(s) from the cleaned data
            jobs = llm.extract_jobs(data)

            for job in jobs:
                skills = job.get('skills', [])
                if not skills:
                    st.warning("No skills found for a job entry. Skipping...")
                    continue

                # Query ChromaDB for relevant links based on extracted skills
                links = portfolio.query_links(skills)

                # Generate cold email using the job description and relevant links
                email = llm.write_mail(job, links)
                st.code(email, language='markdown')

        except Exception as e:
            st.error(f"An Error Occurred: {e}")


# The main execution point of the app
if __name__ == "__main__":
    chain = Chain()  # Initialize your LLM Chain
    portfolio = Portfolio()  # Initialize your Portfolio object
    create_streamlit_app(chain, portfolio, clean_text)  # Run the app
