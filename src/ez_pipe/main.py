from sentence_transformers import SentenceTransformer
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, connections, utility


def ingest(file) -> list:
    '''Ingests a text file, automatically chunks it up into a list of strings
    Returns a list of strings'''
    with open(file, "r") as f:
        file_text = f.read()
    sentences = file_text.split(".")
    return sentences

def embed(sentences, model_name="all-MiniLM-L6-v2") -> list:
    '''Embeds a list of sentences with a specific model.
    Returns a list of dicts'''
    pairings = []
    embedder = SentenceTransformer(model_name)
    for sentence in sentences:
        pair = {
            "embedding": embedder.encode(sentence),
            "sentence": sentence
        }
        pairings.append(pair)
    return pairings

default_milvus_params = {
    "host": "localhost",
    "port": 19530,
    "collection_name": "default_collection"
}

def store_in_milvus(data, sink_params=default_milvus_params, overwrite=True):
    '''Stores the specificied data into Milvus using the specified params.'''
    dimensionality = len(data[0]["embedding"])
    connections.connect(
        host=sink_params["host"],
        port=sink_params["port"]
    )
    if overwrite and utility.has_collection(sink_params["collection_name"]):
        utility.drop_collection(sink_params["collection_name"])
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimensionality),
        FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=1024)
    ]
    schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
    collection = Collection(name=sink_params["collection_name"], schema=schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()

    for entry in data:
        collection.insert(entry)
    
    collection.flush()
    return collection.num_entities

