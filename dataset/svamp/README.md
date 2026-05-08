Expected SVAMP file examples:

- `test.json`
- `test.jsonl`

Supported record shapes include:

```json
{
  "Body": "Tom has 2 apples.",
  "Question": "He buys 3 more. How many apples does he have now?",
  "Answer": 5,
  "Equation": "2+3"
}
```

or

```json
{
  "question": "Tom has 2 apples and buys 3 more. How many apples does he have now?",
  "answer": "5"
}
```
