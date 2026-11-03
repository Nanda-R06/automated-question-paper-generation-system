# Automatic Question Paper Generation System

A Python desktop application that takes a paragraph of text as input, automatically generates questions from it, and evaluates their quality by comparing them against ChatGPT-generated questions using an **F1 score**.

## Features

- **Paragraph input:** paste any passage into the GUI.
- **Automatic question generation:** produces questions from the text, classified by Bloom's Taxonomy level (Remember, Understand, Apply, Analyze, Evaluate, Create).
- **ChatGPT comparison:** compares the generated questions against reference questions produced by ChatGPT for the same paragraph.
- **F1 score evaluation:** reports Precision, Recall and F1 to measure how closely the system matches ChatGPT.
- **Simple GUI:** built with Tkinter, with no web setup required.

## Project Structure

```
MiniProject/
├── gui_app.py            # Tkinter GUI, question generation and F1 evaluation
├── dataset_generator.py  # Builds the Bloom's taxonomy dataset
├── bloom_dataset.csv     # Dataset used for question-type classification
└── README.md
```

## How It Works

1. **Input:** the user enters a paragraph in the GUI.
2. **Generation:** the system extracts key sentences and concepts, then builds questions using Bloom's-taxonomy-based templates.
3. **Reference:** ChatGPT questions for the same paragraph are provided as the reference set.
4. **Similarity matching:** each generated question is compared with the reference questions using word-overlap similarity (stopwords removed). A pair counts as a match if its similarity exceeds a threshold.
5. **Scoring:**

```
Precision = matched generated questions / total generated questions
Recall    = matched reference questions / total reference questions
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

A higher F1 means the system's questions are closer to ChatGPT's.

## Requirements

- Python 3.8+
- Tkinter (included with Python on Windows)
- Standard library only: `re`, `math`, `collections`

## Installation and Usage

```bash
git clone <your-repo-url>
cd MiniProject
python gui_app.py
```

If `python` is not recognized on Windows, use `py gui_app.py`.

1. Paste a paragraph into the input box.
2. Click **Generate Questions**.
3. Paste or load the ChatGPT reference questions.
4. Click **Evaluate** to view Precision, Recall and F1.

## Example Output

```
Generated Questions:
1. What is photosynthesis?
2. Explain how plants convert sunlight into energy.
...

Precision: 0.70
Recall:    0.64
F1 Score:  0.67
```

## Future Improvements

- Use semantic similarity (sentence embeddings) instead of word overlap.
- Fetch ChatGPT reference questions automatically through an API.
- Export the final question paper as PDF or DOCX.
- Add difficulty levels and marks allocation.

## Author

Nanda Rajesh, B.Tech CSE, Vimal Jyothi Engineering College
