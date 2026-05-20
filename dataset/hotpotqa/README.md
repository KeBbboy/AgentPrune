Expected HotpotQA split files in this directory:

- `hotpot_train_v1.1.json` or `train.json`
- `hotpot_dev_distractor_v1.json` or `dev.json`
- optionally `hotpot_dev_fullwiki_v1.json`

Supported record shapes include the official HotpotQA format:

```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "answer": "Arthur's Magazine",
  "supporting_facts": [["Arthur's Magazine", 0], ["First for Women", 0]],
  "context": [
    ["Arthur's Magazine", ["Arthur's Magazine was an American literary periodical published in Philadelphia in the 1840s."]],
    ["First for Women", ["First for Women is a woman's magazine published by Bauer Media Group in the USA."]]
  ],
  "type": "comparison",
  "level": "medium"
}
```
