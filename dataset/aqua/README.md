Expected AQuA split files in this directory:

- `dev.json` or `dev.jsonl`
- `test.json` or `test.jsonl`
- optionally `train.json` / `val.json`

Supported record shapes:

```json
{
  "question": "What is ...?",
  "options": ["A) ...", "B) ...", "C) ...", "D) ...", "E) ..."],
  "correct": "C"
}
```

or

```json
{
  "question": "What is ...?",
  "A": "...",
  "B": "...",
  "C": "...",
  "D": "...",
  "E": "...",
  "answer": "C"
}
```
