import config                                                       # My own keys hiding there ..

ASTRA_DB_SECURE_BUNDLE_PATH = config.MyAstra_DB_secure_bundle_path  #"<<PATH_TO_YOUR_SECURE_BUNDLE>>"
ASTRA_DB_APPLICATION_TOKEN = config.MyAstra_DB_app_token            #"<<YOUR_TOKEN_VALUE>>"       #"AstraCD:..."
ASTRA_DB_CLIENT_ID = config.MyAstra_DB_client_ID                    #"token"
ASTRA_DB_CLIENT_SECRET = config.MyAstra_DB_client_secret
ASTRA_DB_KEYSPACE = config.MyAstra_DB_Keyspace                      #"<<YOUR_KEYSPACE_NAME>>"
OPENAI_API_KEY = config.MyOpenAPIkey                                #"<<YOUR_OPENAI_KEY>>"

from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset

cloud_config = {
    'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
myEmbedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

myCassandraVStore = Cassandra(
    embedding=myEmbedding,
    session=astraSession,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name="qa_mini_demo"
)

print("Loading date from huggingface")
myDataset = load_dataset("Biddls/Onion_News", split="train")
headlines = myDataset["text"][:50]

print("\nGenerating embeddings and storing in AstraDB")
myCassandraVStore.add_texts(headlines)

print("inserted %i headlines.\n" % len(headlines))
