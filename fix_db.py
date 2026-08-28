"""
Remove all QA memory entries that were written by fallback synthesis
(identified by source='memory:' with no route suffix), then fetch
Thomas Edison's biography from the web and store it properly.
"""
import chromadb
import config

client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
col = client.get_collection(
    name=config.CHROMA_KB_COLLECTION,
)

# Fetch all entries
all_data = col.get(include=["documents", "metadatas"])
ids       = all_data["ids"]
docs      = all_data["documents"]
metas     = all_data["metadatas"]

bad_ids = []
kept    = 0
for id_, doc, meta in zip(ids, docs, metas):
    src = meta.get("source", "")
    # "memory:" (no route suffix) means it was written by fallback synthesis — bad
    if src == "memory:":
        print(f"  Removing bad entry: id={id_!r}, source={src!r}")
        print(f"    text={doc[:100]!r}")
        bad_ids.append(id_)
    else:
        kept += 1

if bad_ids:
    col.delete(ids=bad_ids)
    print(f"\nDeleted {len(bad_ids)} bad entries. {kept} clean entries remain.")
else:
    print("No bad entries found.")

# Verify what remains
remaining = col.get(include=["documents", "metadatas"])
print("\n--- Remaining entries ---")
for doc, meta in zip(remaining["documents"], remaining["metadatas"]):
    print(f"  source={meta.get('source','')!r}: {doc[:80]!r}")
