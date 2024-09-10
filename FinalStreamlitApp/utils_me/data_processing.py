import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import streamlit as st

import nltk
import spacy
import os

def download_nltk_data():
    nltk_data = ['punkt', 'vader_lexicon', 'wordnet']
    for data in nltk_data:
        try:
            nltk.data.find(f'tokenizers/{data}')
        except LookupError:
            nltk.download(data, quiet=True)

def download_spacy_model():
    model_name = 'en_core_web_sm'
    if not spacy.util.is_package(model_name):
        spacy.cli.download(model_name, False, '--quiet')

# Call the download functions
download_nltk_data()
download_spacy_model()

#python -m spacy download en
from bs4 import BeautifulSoup

def remove_specific_divs(html_content):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Find all div tags with the class "stCodeBlock"
    for div in soup.find_all("div", class_="stCodeBlock"):
        div.decompose()  # Remove the div tag from the tree

    # Return the modified HTML as a string
    return str(soup)

def pre_process_text(text):
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Convert text to lowercase
    text = text.lower()

    # Remove timestamps, ensuring spaces remain intact
    text = re.sub(r'\(\d{2}:\d{2}\)', ' ', text)

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])

    # Remove words with less than 2 characters (optional, and should be used with caution)
    # text = ' '.join([word for word in text.split() if len(word) > 2])

    # Lemmatize words
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    # Remove words with less than 2 characters after lemmatization (if still needed)
    text = ' '.join([word for word in text.split() if len(word) > 2])

    # Ensure spaces are preserved after certain operations
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def perform_wordcloud(transcript):
    from PIL import Image
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from io import BytesIO
    """This function performs the following analysis on the transcript:
        i. Word count (top words; two and three word phrases)
        ii. Word cloud
        iii. Entity identification and sentiment analysis
        iv. Topic identification and sentiment analysis
        v. Sentiment analysis for sentences
    """
    # Pre-process the text
    text = pre_process_text(transcript)
    
    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=800, background_color='white', max_words=150).generate(text)
    
    # Save the word cloud image to a BytesIO object
    buffer = BytesIO()
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    # Save the plot to the buffer
    plt.savefig(buffer, format='png')
    plt.close()
    
    # Seek to the beginning of the buffer
    buffer.seek(0)
    
    # Open the image from the buffer
    img = Image.open(buffer)
    
    return img


def perform_and_show_NER(transcript):
    from transformers import AutoTokenizer, AutoModelForTokenClassification, TFAutoModelForTokenClassification
    from transformers import pipeline

    tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
    model = TFAutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    import pandas as pd
    import spacy

    # Load the English language model
    nlp = spacy.load("en_core_web_sm")


    # Perform NER
    doc = nlp(transcript)

    # # Print the entities
    # for ent in doc.ents:
    #     print(f"Entity: {ent.text}, Label: {ent.label_}")
 
    # Optional: Count entity types
    entity_counts = {}
    for ent in doc.ents:
        if ent.label_ in entity_counts:
            entity_counts[ent.label_] += 1
        else:
            entity_counts[ent.label_] = 1

    # print("\nEntity type counts:")
    # for entity_type, count in entity_counts.items():
    #     print(f"{entity_type}: {count}")
    from IPython.display import display, HTML
    from spacy import displacy
    html = displacy.render(doc, style="ent", page=True)

    # Display the visualization
    # st_html = HTML(html)
    fig = top_entities_ner(entity_counts, top_n=10)
    return html, fig



def top_entities_ner(entity_counts, top_n=10, output_filename='top_entities_graph_plotly.png'):
    import plotly.express as px
    import pandas as pd
    """
    Plots the top N named entities from a given entity count dictionary using Plotly.

    Args:
    - entity_counts (dict): A dictionary with entity types as keys and their counts as values.
    - top_n (int): Number of top entities to display in the plot. Default is 10.

    Returns:
    - fig (plotly.graph_objs._figure.Figure): The Plotly figure object.    
    """
    
    # Convert entity_counts to a DataFrame
    df = pd.DataFrame(list(entity_counts.items()), columns=['Entity', 'Count'])

    # Sort by Count in descending order and select top N entities
    df_top = df.nlargest(top_n, 'Count')

    # Create the bar plot using Plotly
    fig = px.bar(
        df_top,
        x='Entity',
        y='Count',
        text='Count',
        color='Entity',  # This adds color based on the entity type
        color_discrete_sequence=px.colors.qualitative.Pastel,  # Custom color palette
        title=f'Top {top_n} Named Entities mentioned in the Transcript'
    )

    # Customize the layout
    fig.update_layout(
        title_font_size=16,
        xaxis_title='Entity Types',
        yaxis_title='Count',
        xaxis_tickangle=-45,
        xaxis_tickfont_size=12,
        yaxis_tickfont_size=12,
        title_pad=dict(t=20, b=20),
        margin=dict(t=50, b=50),
    )


    # Adjust the text position to be above the bars
    fig.update_traces(textposition='outside', textfont=dict(size=12))

    # Return the figure object
    return fig





def top_persons_and_org(transcript, top_n=5):
    import spacy
    from collections import Counter
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Load the English language model
    nlp = spacy.load("en_core_web_sm")

    # Create a mapping for normalization
    entity_mapping = {
        "clinton": "Hillary Clinton",
        "hillary clinton": "Hillary Clinton",
        "hillary": "Hillary Clinton",
        "trump": "Donald Trump",
        "donald trump": "Donald Trump",
        "donald": "Donald Trump",
        "americans": "American",
        "american": "American",
        "republicans": "Republicans",
        "african-americans": "African-Americans",
    }

    # Perform NER
    doc = nlp(transcript)

    # Dictionaries to hold entity counts
    persons_entity = Counter()
    combined_org_norp_entity = Counter()

    # Count and normalize entities
    for ent in doc.ents:
        normalized_text = entity_mapping.get(ent.text.lower(), ent.text)
        if ent.label_ == "PERSON":
            persons_entity[normalized_text] += 1
        elif ent.label_ in ["ORG", "NORP"]:
            combined_org_norp_entity[normalized_text] += 1

    # Get the top N entities for each category
    top_n_persons = persons_entity.most_common(top_n)
    top_n_combined_org_norp = combined_org_norp_entity.most_common(top_n)

    # Prepare data for plotting
    entities = [person[0] for person in top_n_persons] + [org[0] for org in top_n_combined_org_norp]
    person_counts = [person[1] for person in top_n_persons] + [0] * top_n
    org_norp_counts = [0] * top_n + [org[1] for org in top_n_combined_org_norp]

    # Create subplot
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # Add traces for each category
    fig.add_trace(
        go.Bar(x=entities, y=person_counts, name="PERSON", marker_color='skyblue'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=entities, y=org_norp_counts, name="ORG/NORP", marker_color='lightgreen'),
        row=1, col=1
    )

    # Update layout
    fig.update_layout(
        title_text=f"Top {top_n} Entities Comparison",
        barmode='group',
        yaxis_title="Counts",
        xaxis_title="Entities",
        legend_title="Entity Type",
        font=dict(size=12),
        height=600,
        width=1000
    )

    # Update x-axis
    fig.update_xaxes(tickangle=45)

    # Print the results (optional)
    st.subheader(f"Top {top_n} PERSON entities with counts:")
    for person, count in top_n_persons:
        st.write(f"{person}: {count}")

    st.subheader(f"\nTop {top_n} combined ORG and NORP entities with counts:")
    for entity, count in top_n_combined_org_norp:
        st.write(f"{entity}: {count}")

    # Return the figure object for further use or visualization
    return fig


import re
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data once when the module is loaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

def process_sentiment_analysis_sentence(transcript):
    # Lowercase the transcript and remove timestamps
    transcript = transcript.lower()
    transcript = re.sub(r'\(\d{2}:\d{2}\)', '', transcript)

    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Tokenize the transcript into sentences
    sentences = sent_tokenize(transcript)

    # Group sentences into chunks of 3
    grouped_sentences = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]

    # Perform sentiment analysis on each group of 3 sentences
    sentiments = []
    for group in grouped_sentences:
        sentiment = sia.polarity_scores(group)
        sentiments.append({
            'sentence_group': group,
            'sentiment': sentiment,
            'overall': 'positive' if sentiment['compound'] > 0 else 'negative' if sentiment['compound'] < 0 else 'neutral'
        })

    # Convert the results to a DataFrame for easy viewing
    results_df = pd.DataFrame(sentiments)

    # Return results_df for further processing or display
    return results_df

def most_positive_n_negative(text):
    # Perform sentiment analysis
    results_df = process_sentiment_analysis_sentence(text)
    # drop column sentiment in results df


    st.write("## Sentiment Summary")
    st.write(results_df['overall'].value_counts(normalize=True))

    # Display the most positive and most negative sentence groups
    most_positive = results_df.loc[results_df['sentiment'].apply(lambda x: x['compound']).idxmax()]
    most_negative = results_df.loc[results_df['sentiment'].apply(lambda x: x['compound']).idxmin()]

    st.write("### Most Positive Sentence Group")
    # compound is negative, print negative, if greater than 0, print positive
    compound = most_positive['sentiment']["compound"]
    if compound > 0:
        st.write(most_positive['sentence_group'])

        st.write("Positive with compound score: ", compound)

    # st.write(f"Sentiment: {most_positive['sentiment']["compound"]}")

    compound = most_negative['sentiment']["compound"]
    if compound < 0:
        st.write(most_negative['sentence_group'])
        st.write("Negative with compound score: ", compound)
    # st.write("### Most Negative Sentence Group")
    # st.write(most_negative['sentence_group'])
    # st.write(f"Sentiment: {most_negative['sentiment']}")

    # Optionally, display the detailed results
    st.write("### Detailed Sentiment Analysis Results")
    # drop column sentiment in results df
    st.dataframe(results_df.drop(columns='sentiment'))
import spacy

# Load the English language model for NER
nlp = spacy.load("en_core_web_sm")

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# def perform_and_show_NER_with_sentiment(transcript):
#     """
#     This function extracts named entities from the transcript and performs sentiment analysis on sentences
#     mentioning each entity. It returns a DataFrame with the entity, type, sentiment, and count.
#     """
#     # Preprocess transcript
#     transcript = re.sub(r'\(\d{2}:\d{2}\)', '', transcript)

#     # Perform NER
#     doc = nlp(transcript)

#     # Tokenize transcript into sentences
#     sentences = sent_tokenize(transcript)

#     # Store sentiment analysis for each entity
#     entity_sentiment_data = []

#     # Loop over entities found by spacy
#     for ent in doc.ents:
#         entity_text = ent.text
#         entity_type = ent.label_

#         # Find sentences mentioning the entity
#         entity_sentences = [sentence for sentence in sentences if entity_text in sentence]

#         # Perform sentiment analysis on those sentences
#         for sentence in entity_sentences:
#             sentiment = sia.polarity_scores(sentence)
#             overall_sentiment = 'positive' if sentiment['compound'] > 0 else 'negative' if sentiment['compound'] < 0 else 'neutral'
#             entity_sentiment_data.append({
#                 'entity': entity_text,
#                 'entity_type': entity_type,
#                 'sentiment': overall_sentiment
#             })

#     # Create DataFrame from the data
#     entity_sentiment_df = pd.DataFrame(entity_sentiment_data)

#     return entity_sentiment_df

def perform_and_show_NER_with_sentiment(transcript):
    """
    This function extracts named entities from the transcript and performs sentiment analysis on sentences
    mentioning each entity. It returns a DataFrame with the entity, type, sentiment, and count.
    """

    # Define the entity mapping for normalization (includes Joe Biden and other mappings for Trump analysis)
    entity_mapping = {
        "joe": "Joe Biden",
        "biden": "Joe Biden",
        "joe biden": "Joe Biden",
        "trump": "Donald Trump",
        "donald trump": "Donald Trump",
        "donald": "Donald Trump",
        "americans": "American",
        "american": "American",
        "republicans": "Republicans",
        "african-americans": "African-Americans",
        "clinton": "Hillary Clinton",
        "hillary clinton": "Hillary Clinton",
        "hillary": "Hillary Clinton"
    }

    # Preprocess transcript: remove timestamps and unnecessary details
    transcript = re.sub(r'\(\d{2}:\d{2}\)', '', transcript)

    # Perform NER using Spacy
    doc = nlp(transcript)

    # Tokenize transcript into sentences
    sentences = sent_tokenize(transcript)

    # Store sentiment analysis for each entity
    entity_sentiment_data = []

    # Loop over entities found by Spacy
    for ent in doc.ents:
        # Normalize the entity text using the entity mapping
        normalized_entity = entity_mapping.get(ent.text.lower(), ent.text)
        entity_type = ent.label_

        # Find sentences mentioning the entity
        entity_sentences = [sentence for sentence in sentences if normalized_entity in sentence]

        # Perform sentiment analysis on those sentences
        for sentence in entity_sentences:
            sentiment = sia.polarity_scores(sentence)
            overall_sentiment = 'positive' if sentiment['compound'] > 0 else 'negative' if sentiment['compound'] < 0 else 'neutral'
            entity_sentiment_data.append({
                'entity': normalized_entity,
                'entity_type': entity_type,
                'sentiment': overall_sentiment
            })

    # Create a DataFrame from the entity sentiment data
    entity_sentiment_df = pd.DataFrame(entity_sentiment_data)

    return entity_sentiment_df
import plotly.express as px

# def visualize_sentiment_distribution_for_top_entities(entity_sentiment_df, top_n=2):
#     """
#     Visualizes the sentiment distribution for the top N named entities.
#     This includes applying entity mapping for normalization, and generating pie charts.
#     """
    
#     # Define the entity mapping for normalization
#     entity_mapping = {
#         "clinton": "Hillary Clinton",
#         "hillary clinton": "Hillary Clinton",
#         "hillary": "Hillary Clinton",
#         "trump": "Donald Trump",
#         "donald trump": "Donald Trump",
#         "donald": "Donald Trump",
#         "americans": "American",
#         "american": "American",
#         "republicans": "Republicans",
#         "african-americans": "African-Americans",
#     }

#     # Apply entity mapping to normalize the entity names
#     entity_sentiment_df['entity'] = entity_sentiment_df['entity'].apply(
#         lambda entity_text: entity_mapping.get(entity_text.lower(), entity_text)
#     )

#     # Group the sentiment data by entity and sentiment, and count occurrences
#     entity_sentiment_counts = entity_sentiment_df.groupby(['entity', 'sentiment']).size().reset_index(name='count')

#     # Get the total counts for each entity to determine the top N entities
#     entity_total_counts = entity_sentiment_df['entity'].value_counts()

#     # Select the top N entities based on their occurrence in the text
#     top_entities = entity_total_counts.head(top_n).index.tolist()

#     # Loop over each entity to generate pie charts (only for top N entities)
#     for entity in top_entities:
#         entity_data = entity_sentiment_counts[entity_sentiment_counts['entity'] == entity]

#         # Create a pie chart for the current entity
#         fig = px.pie(entity_data, values='count', names='sentiment', 
#                      title=f'Sentiment Distribution for {entity}',
#                      color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96'])  # Custom color palette

#         # Display the pie chart in Streamlit
#         st.plotly_chart(fig)

def visualize_sentiment_distribution_for_top_entities(entity_sentiment_df, top_n=2):
    """
    Visualizes the sentiment distribution for the top N named entities.
    This includes applying entity mapping for normalization, and generating pie charts.
    """
    entity_mapping = {
        "joe": "Joe Biden",
        "biden": "Joe Biden",
        "joe biden": "Joe Biden",
        "trump": "Donald Trump",
        "donald trump": "Donald Trump",
        "donald": "Donald Trump",
        "americans": "American",
        "american": "American",
        "republicans": "Republicans",
        "african-americans": "African-Americans",
        "clinton": "Hillary Clinton",
        "hillary clinton": "Hillary Clinton",
        "hillary": "Hillary Clinton"
    }

    # Apply entity mapping to normalize the entity names (e.g., "Joe Biden" and "Donald Trump")
    entity_sentiment_df['entity'] = entity_sentiment_df['entity'].apply(
        lambda entity_text: entity_mapping.get(entity_text.lower(), entity_text)
    )

    # Group the sentiment data by entity and sentiment, and count occurrences
    entity_sentiment_counts = entity_sentiment_df.groupby(['entity', 'sentiment']).size().reset_index(name='count')

    # Get the total counts for each entity to determine the top N entities
    entity_total_counts = entity_sentiment_df['entity'].value_counts()

    # Select the top N entities based on their occurrence in the text
    top_entities = entity_total_counts.head(top_n).index.tolist()

    # Loop over each entity to generate pie charts (only for top N entities)
    for entity in top_entities:
        entity_data = entity_sentiment_counts[entity_sentiment_counts['entity'] == entity]

        # Create a pie chart for the current entity
        fig = px.pie(entity_data, values='count', names='sentiment',
                     title=f'Sentiment Distribution for {entity}',
                     color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96'])  # Custom color palette

        # Display the pie chart in Streamlit
        st.plotly_chart(fig)