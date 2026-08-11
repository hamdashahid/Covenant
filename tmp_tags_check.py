import os
import tempfile
from persistence.sqlite_store import SQLiteStore

with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
    path = tmp.name
store = SQLiteStore(db_path=path)
store.create_session('sess-tags-check', 'model')
store.merge_session_tags('sess-tags-check', ['ciap-ready', 'existing'])
store.merge_session_tags('sess-tags-check', ['ciap-ready', 'new-tag'])
print(store.get_session('sess-tags-check')['tags'])
os.unlink(path)
