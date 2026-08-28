"""
Inspect all collections and show what's in them,
then delete only entries whose source contains 'memory' (qa_memory write-backs).
"""
import chromadb
import config

client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
collections = client.list_collections()
print(f"Collections found: {[c.name for c in collections]}\n")

for col_meta in collections:
    col = client.get_collection(col_meta.name)
    count = col.count()
    print(f"=== {col_meta.name} ({count} docs) ===")
    if count == 0:
        print("  (empty)")
        continue
    results = col.get(include=["documents", "metadatas"])
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        src = meta.get("source", "")
        print(f"  [{i}] source={src!r}")
        print(f"       text={doc[:120]!r}")
    print()
