bash
bash
bash
# 🧠 LLM QA Evaluation App with LangSmith

This app provides a modular workflow for evaluating Question-Answering (QA) chains using OpenAI's GPT-4 and [LangSmith](https://smith.langchain.com). It enables you to ingest datasets, run LLM evaluations, and view analytics—all with clear separation of concerns.

---

## 🚀 Key Features

- **Dataset Ingestion:** Easily reset and populate your QA dataset for repeatable, reliable testing.
- **LLM Evaluation:** Run LLMs on your dataset and score outputs using LangSmith's `qa` evaluator (LLM-as-a-Judge).
- **Analytics & Debugging:** Get instant feedback on run success, errors, and sample results in your terminal, plus full trace and evaluation in the LangSmith UI.
- **Modular Design:** Clean separation between data ingestion, evaluation logic, and app entry point for easy extension and maintenance.

---

## 🧱 Tech Stack

- [LangChain](https://www.langchain.com/)
- [OpenAI GPT-4](https://platform.openai.com/)
- [LangSmith](https://smith.langchain.com/)
- Python 3.10+

---

## 📂 Project Structure

```text
main.py             # Entry point: runs data ingestion and evaluation
llm_evaluator.py    # LLMRunEvaluator class: builds chain, runs evaluation, prints analytics
data_ingestor.py    # DatasetIngestor class: resets and populates dataset
README.md           # You're reading it!
```

---

## ⚡️ Quickstart

1. **Install dependencies:**
   ```bash
   pip install langchain langsmith openai
   ```

2. **Set your environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export LANGSMITH_API_KEY="your-langsmith-key"
   export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
   export LANGSMITH_PROJECT="pr-large-ladybug-30"
   ```

3. **Run the app:**
   ```bash
   python main.py
   ```

---

## 📊 Viewing Results in LangSmith

- **Project Dashboard:** https://smith.langchain.com
- **Your Dataset:** Find "Trivia QA" under the "Datasets" section
- **Evaluation Results:** Go to your dataset > "Compare" or "Evaluations" tab
- **Run Traces & Errors:** Projects > [Your Project] > Runs (filter by error for debugging)

---

## 🧠 Example Questions Evaluated

- What is the capital of Germany? → Berlin
- Who wrote Hamlet? → William Shakespeare
- What is the speed of light in vacuum? → 299,792,458 m/s

