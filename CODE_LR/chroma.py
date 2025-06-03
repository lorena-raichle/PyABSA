import chromadb


def create_chroma_db():
    # Create a ChromaDB client (persistent storage)
    client = chromadb.PersistentClient(path="../chroma_db")

    # Alternative: In-memory client (data won't persist)
    # client = chromadb.Client()

    # Create or get a collection
    collection = client.get_or_create_collection(
        name="my_collection",
        metadata={"description": "A sample collection"}
    )

    # Sample documents to add
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a versatile programming language.",
        "ChromaDB is a vector database for AI applications.",
        "Machine learning models require large datasets.",
        "Der Strand in Sevilla war wunderschön und das Wasser war kalt"
    ]

    # Sample metadata (optional)
    metadatas = [
        {"category": "animals", "source": "example"},
        {"category": "programming", "source": "example"},
        {"category": "database", "source": "example"},
        {"category": "ai", "source": "example"},
        {"category": "nlp", "source": "example"}
    ]

    # Sample IDs for each document
    ids = [f"doc_{i}" for i in range(len(documents))]

    # Add documents to the collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Created collection '{collection.name}' with {collection.count()} documents")

    results = collection.query(
        query_texts=["Strand und meer"],
        n_results=2
    )

    print("\nQuery results for 'programming languages':")
    for i, doc in enumerate(results['documents'][0]):
        print(f"{i + 1}. {doc}")

        print(f"   Distance: {results['distances'][0][i]:.4f}")

    return client, collection


def list_collections(client):
    """List all collections in the database"""
    collections = client.list_collections()
    print(f"\nAvailable collections: {len(collections)}")
    for collection in collections:

        print(f"- {collection.name}: {collection.count()} documents")

if __name__ == "__main__":
    # Create the database and collection
    client, collection = create_chroma_db()

    # List all collections
    list_collections(client)

    # Get collection info
    print(f"\nCollection metadata: {collection.metadata}")
    print(f"Total documents in collection: {collection.count()}")

