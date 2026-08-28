"""Clear only the QA memory and conversation memory collections (keeps knowledge_base)."""
import chromadb
import config

client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
existing = {col.name for col in client.list_collections()}

for name in [config.CHROMA_QA_MEMORY_COLLECTION, config.CHROMA_CONV_MEMORY_COLLECTION]:
    if name in existing:
        col = client.get_collection(name)
        count = col.count()
        client.delete_collection(name)
        print(f'Cleared "{name}" ({count} entries removed)')
    else:
        print(f'"{name}" does not exist — skipping')

print("Done.")
