# SpecAssist

## Reindexing Excel data

If you need to rebuild the local database after importer changes, delete the old DB and reindex:

```bash
rm data/app.db && python main.py --reindex simpe1.xlsx
```

If you are using the bundled sample file from this repo, run:

```bash
rm data/app.db && python main.py --reindex base/simpe1.xlsx
```
