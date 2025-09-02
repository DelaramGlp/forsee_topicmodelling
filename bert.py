from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction import text
import umap.umap_ as umap
#import spacy

import os

from wordcloud import WordCloud
import csv
import pandas as pd
import plotly.io as pio
import plotly.offline as pyo
import matplotlib.pyplot as plt

# Set Plotly renderer
pio.renderers.default = 'browser'

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_wordcloud(model, topic):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    return wc.generate_from_frequencies(text)
    #plt.imshow(wc, interpolation="bilinear")
    #plt.axis("off")
    #plt.show()


stop_words = list(text.ENGLISH_STOP_WORDS.union(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
                                                 "spring",
                "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii",
                "oj", "en", "http", "tfeu", "isbn", "pdf",
                "ai", "system",
                "europe", "eu", "ec", "eli", "commission", "union", "european", "parliament", "council", "directive", "regulation", "law", "act", "acts","hleg",
                "article", "annex", "chapter", "point", "section", "text", "footnote", "introduction", "appendix",
              "based"]))

file_path = 'doc.csv'
file_name = os.path.splitext(os.path.basename(file_path))[0]
df = pd.read_csv(file_path)



#nlp = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])



#embedding_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") #appropiate for Semantic similarity/clustering, all-MiniLM-L6-v2 for fast, light-weight stuff and use all-mpnet-base-v2 for better accuracy
umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42) #reduce the dimensions in the embedding
hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=5) # clustring model, added it for shorter documents
vectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words= stop_words)
representation_model = MaximalMarginalRelevance(diversity=0.3) #because we have no lemmatizers

topic_model = BERTopic(embedding_model = embedding_model, umap_model = umap_model, vectorizer_model = vectorizer_model, hdbscan_model = hdbscan_model, representation_model = representation_model)

docs = df.Header.to_list()
 

topics, probabilities = topic_model.fit_transform(docs)


topic_info = topic_model.get_topic_info()
topic_info.to_csv(file_name+"_topic_info.csv", index=False)
print(topic_info)


#print(model.get_topic(-1))
#print(model.get_topic(0))
#print(model.get_topic(1))


fig=topic_model.visualize_barchart(top_n_topics=len(topic_model.get_topics()))
pyo.plot(fig, filename= file_name+'_bert_barchart.html')


'''
fig = model.visualize_topics()
fig.show()
'''

# Similarity Matrix
'''
fig = model.visualize_heatmap() #it might not be very informative if there are too many topics identified
fig.show()
'''


# Hierarchical Topics
'''
hierarchical_topics = model.hierarchical_topics(docs) 
fig = model.visualize_hierarchy(hierarchical_topics=hierarchical_topics) #probabilities of a certain document falling under each topic - as we don't run the model over mutiple documents this might not be very useful
fig.show()
'''

#Topics over time - if there is a time element to be included 
'''
topics_over_time = model.topics_over_time(docs, topics, timestamp, nr_bins=20)
model.visualize_topics_over_time(topics_over_time, top_n_topics=20) 
'''

# Show wordcloud
#for topic_id in topic_info.Topic:
#    create_wordcloud(model, topic_id)


topic_ids = list(topic_info.Topic)

# Grid dimensions
cols = 5
rows = (len(topic_ids) + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
axes = axes.flatten()

for i, topic_id in enumerate(topic_ids):
    wc = create_wordcloud(topic_model, topic_id)
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].axis('off')
    axes[i].set_title(f"Topic {topic_id}", fontsize=14)

# Turn off unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
#plt.show()
plt.savefig(file_name+"_bert_wordcloud.svg")


#fig = topic_model.visualize_term_rank()
#fig.show()


