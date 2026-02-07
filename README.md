# SpecAssist

## Offline web app

Для автономного офлайн-использования откройте `dist/index.html`. Подробная инструкция — в [README_WEB.md](README_WEB.md).

## Reindexing Excel data

If you need to rebuild the local database after importer changes, delete the old DB and reindex:

```bash
rm data/app.db && python main.py --reindex simpe1.xlsx
```

If you are using the bundled sample file from this repo, run:

```bash
rm data/app.db && python main.py --reindex base/simpe1.xlsx
```
