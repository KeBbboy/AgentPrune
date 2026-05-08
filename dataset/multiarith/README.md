Expected MultiArith file examples:

- `test.json`
- `test.jsonl`

Supported record shapes include:

```json
{
  "question": "Tom has 2 apples and buys 3 more. How many apples does he have?",
  "answer": "5"
}
```

or

```json
{
  "sQuestion": "Tom has 2 apples and buys 3 more. How many apples does he have?",
  "lSolutions": [5],
  "lEquations": ["2+3=5"]
}
```
