# 🧠 LLM QA Evaluation App with LangSmith

This app provides a modular workflow for evaluating Question-Answering (QA) chains using OpenAI's GPT-4 and [LangSmith](https://smith.langchain.com). It enables you to ingest datasets, run LLM evaluations, and view analytics—all with clear separation of concerns.

---

This app is fully instrumented with **LangSmith**, enabling:

| Feature                      | What You Get                                                  | Where to Find in LangSmith UI |
|-----------------------------|---------------------------------------------------------------|--------------------------------|
| **1. Tracing & Debugging**   | View full trace of each run (input, output, prompt, time)     | `Projects > [Your Project] > Runs` |
| **2. Evaluation (LLM-as-Judge)** | Automated correctness scoring using GPT-4                  | `Datasets > [Your Dataset] > Evaluations` |
| **3. Dataset Management**    | Manage inputs & expected outputs for repeatable testing       | `Datasets > Trivia QA` |
| **4. Model Comparison**      | Compare outputs across models or versions                    | `Datasets > Compare` |
| **5. Error Monitoring**      | Catch failed runs and view error messages                     | `Projects > [Your Project] > Runs > Filter by Error` |
| **6. Performance Monitoring**| View latency and runtime statistics                           | `Runs Table > Execution Time Column` |
| **7. Team Sharing**          | Share dashboards with collaborators                           | `Share` buttons in datasets, projects, or runs |


It provides crucial info regarding **Latency**, **Evaluation Results** and **Token usuage**.
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

## 🖥️ Dashboard Screenshots
![Alt text](https://github.com/ArianFotouhi/llmPostDeployment-EvaluationAndMonitoring/blob/Langsmith/assets/1-langsmith.png)
![Alt text](https://github.com/ArianFotouhi/llmPostDeployment-EvaluationAndMonitoring/blob/Langsmith/assets/2-langsmith.png)
![Alt text](https://github.com/ArianFotouhi/llmPostDeployment-EvaluationAndMonitoring/blob/Langsmith/assets/3-langsmith.png) 

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

