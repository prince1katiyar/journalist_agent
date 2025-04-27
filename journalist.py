
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from newspaper import Article
from duckduckgo_search import DDGS
from urllib.parse import urlparse
import logging
import os
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Streamlit App setup
st.set_page_config(page_title="AI Journalist Agent", page_icon="🗞️", layout="wide")
st.title("🗞️ AI Journalist Agent")
st.caption("Automatically research, analyze, summarize, and write high-quality articles using GPT-4o or GPT-3.5-turbo.")



# Session state initialization for tracking progress
if "current_step" not in st.session_state:
    st.session_state.current_step = "Not Started"
if "research_results" not in st.session_state:
    st.session_state.research_results = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# User inputs
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    model_choice = st.selectbox(
        "Select AI Model", 
        ["gpt-3.5-turbo (Fast)", "gpt-4o (High Quality)"],
        help="GPT-3.5-turbo is faster but GPT-4o produces higher quality articles"
    )
    
    article_length = st.slider(
        "Article Length (words)", 
        min_value=300, 
        max_value=1000, 
        value=500, 
        step=100,
        help="Longer articles require more processing time"
    )
    
    num_sources = st.slider(
        "Number of Sources",
        min_value=2,
        max_value=5,
        value=3,
        help="More sources mean more comprehensive research but slower processing"
    )
    
    show_intermediates = st.checkbox("Show Intermediate Results", value=False)
    
    st.subheader("Optional")
    article_style = st.selectbox(
        "Article Style",
        ["Informative", "Persuasive", "Narrative", "Analytical", "Conversational"],
        index=0
    )
    
    target_audience = st.text_input("Target Audience (Optional)", 
                                   placeholder="e.g., General public, Students, Professionals")


# DuckDuckGo search tool
def duckduckgo_search(query):
    try:
        with DDGS() as ddg:
            results = list(ddg.text(query, max_results=num_sources))
        urls = [result["href"] for result in results if result.get("href")]
        return urls if urls else ["No valid search results found."]
    except Exception as e:
        logger.error(f"Error in DuckDuckGo search: {str(e)}")
        return [f"Error performing search: {str(e)}"]

search_tool = Tool(
    name="DuckDuckGoSearch",
    func=duckduckgo_search,
    description="Perform DuckDuckGo searches and retrieve top URLs."
)

# Newspaper3k tool for fetching article text
def fetch_article(url):
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return f"Invalid URL format: {url}"
        article = Article(url, timeout=10)
        article.download()
        article.parse()
        content = article.text
        # Limit content size for faster processing
        return content[:2500] if content and len(content) > 50 else f"Insufficient content from {url}"
    except Exception as e:
        logger.error(f"Error fetching article from {url}: {str(e)}")
        return f"Error fetching content from {url}: {str(e)}"

fetch_tool = Tool(
    name="FetchArticle",
    func=fetch_article,
    description="Fetch article text from URLs with robust error handling."
)

# Set up the main content area
topic = st.text_input("Enter the topic you want an article on:")

# Create placeholder for detailed progress updates
progress_placeholder = st.empty()
message_placeholder = st.empty()
research_placeholder = st.empty()
analysis_placeholder = st.empty()
final_article_placeholder = st.empty()

def update_progress_message(step_name, message, progress_value):
    """Update progress bar and message"""
    progress_bar = progress_placeholder.progress(progress_value)
    message_placeholder.info(f"**Current Step:** {step_name} - {message}")
    st.session_state.current_step = step_name
    # Simulate some processing time to show progress
    time.sleep(0.5)
    return progress_bar

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    try:
        # Extract the model name from selection
        chosen_model = "gpt-3.5-turbo" if "3.5" in model_choice else "gpt-4o"
        
        # Initialize the LLM with the chosen model
        llm = ChatOpenAI(
            api_key=openai_api_key, 
            model=chosen_model,
            temperature=0.7
        )
        
        # Create the journalist agent
        journalist_agent = Agent(
            role="AI Journalist",
            goal=f"Create a {article_style.lower()} article for {target_audience if target_audience else 'a general audience'}",
            tools=[search_tool, fetch_tool],
            llm=llm,
            backstory="A versatile journalist who creates quality articles adapted to different audiences and styles.",
            verbose=True
        )

        if topic and st.button("Generate Article"):
            # Create a container for the full process
            with st.container():
                # Initialize progress tracking
                update_progress_message("Starting", "Initializing article generation process...", 0)
                
                # Step 1: Research
                research_task = Task(
                    description=f"Find {num_sources} relevant and authoritative URLs on '{topic}'. Format as numbered markdown links.",
                    agent=journalist_agent,
                    expected_output=f"{num_sources} markdown-formatted URLs"
                )
                
                update_progress_message("Research", "Searching for relevant sources...", 15)
                
                # Execute research task
                crew_research = Crew(
                    agents=[journalist_agent],
                    tasks=[research_task],
                    verbose=True,
                    process=Process.sequential
                )
                research_result = crew_research.kickoff()
                
                # Process and display research results
                if hasattr(research_result, 'raw'):
                    research_text = str(research_result.raw)
                else:
                    research_text = str(research_result)
                
                st.session_state.research_results = research_text
                
                update_progress_message("Research", "Found relevant sources!", 30)
                
                if show_intermediates:
                    research_placeholder.success("Research Results:")
                    research_placeholder.markdown(research_text)
                
                # Step 2: Analysis
                analysis_task = Task(
                    description=f"Fetch and analyze the content from each URL related to '{topic}'. Summarize each source in 2-3 sentences.",
                    agent=journalist_agent,
                    context=[research_task],
                    expected_output="Concise summaries of each source's key points"
                )
                
                update_progress_message("Analysis", "Analyzing content from sources...", 45)
                
                # Execute analysis task
                crew_analysis = Crew(
                    agents=[journalist_agent],
                    tasks=[analysis_task],
                    verbose=True,
                    process=Process.sequential
                )
                analysis_result = crew_analysis.kickoff()
                
                # Process and display analysis results
                if hasattr(analysis_result, 'raw'):
                    analysis_text = str(analysis_result.raw)
                else:
                    analysis_text = str(analysis_result)
                
                st.session_state.analysis_results = analysis_text
                
                update_progress_message("Analysis", "Source analysis complete!", 60)
                
                if show_intermediates:
                    analysis_placeholder.success("Analysis Results:")
                    analysis_placeholder.markdown(analysis_text)
                
                # Step 3: Writing
                update_progress_message("Writing", "Crafting your article...", 75)
                
                writing_task = Task(
                    description=f"""
                    Write a {article_style.lower()} article about '{topic}' that is approximately {article_length} words. 
                    Target audience: {target_audience if target_audience else 'general readers'}.
                    Format with markdown: include a compelling headline, introduction, main content sections with subheadings, and a conclusion.
                    Base your article on the research and analysis results provided.
                    """,
                    agent=journalist_agent,
                    context=[research_task, analysis_task],
                    expected_output="A well-structured markdown article"
                )
                
                # Execute writing task
                crew_writing = Crew(
                    agents=[journalist_agent],
                    tasks=[writing_task],
                    verbose=True,
                    process=Process.sequential
                )
                writing_result = crew_writing.kickoff()
                
                # Process and display final article
                if hasattr(writing_result, 'raw'):
                    article_text = str(writing_result.raw)
                else:
                    article_text = str(writing_result)
                
                update_progress_message("Complete", "Article generation completed successfully!", 100)
                
                # Display final article
                final_article_placeholder.subheader("📰 Final Article")
                final_article_placeholder.markdown(article_text)
                
                # Add download button
                st.download_button(
                    label="Download Article",
                    data=article_text,
                    file_name=f"{topic.replace(' ', '_')}_article.md",
                    mime="text/markdown"
                )
                
                # Add feedback options
                st.subheader("Article Feedback")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍 Great Article!"):
                        st.success("Thanks for your positive feedback!")
                with col2:
                    if st.button("👌 Good but could be better"):
                        st.info("Thanks for your feedback! What could be improved?")
                        st.text_area("Suggestions for improvement:")
                with col3:
                    if st.button("👎 Needs improvement"):
                        st.warning("We appreciate your honest feedback!")
                        st.text_area("What could be improved?")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"An error occurred: {str(e)}")
else:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")

# Add footer with app info
st.markdown("---")
st.caption("AI Journalist- Powered by CrewAI and OpenAI by Euron ")