# 📊 llmPostDeployment-EvaluationAndMonitoring

**End-to-end framework for evaluating and monitoring LLM responses post-deployment using Retrieval-Augmented Generation (RAG), OpenAI models, Pinecone, and Arize Phoenix.**

---

## 🚀 Overview

This project provides a robust pipeline to:
- Retrieve reference knowledge using Pinecone RAG
- Generate model responses via OpenAI
- Evaluate response quality using Arize Phoenix
- Score and label responses for QA, monitoring, and continuous improvement

---

## 📁 Project Structure

```

llmPostDeployment-EvaluationAndMonitoring/
├── config.py                       # To handle the API keys
├── main.py                        # Core RAG + Evaluation pipeline (class-based)
├── test\_main.py                   # Unit tests for pipeline
├── vectorStore/
│   ├── vector\_store.py            # Pinecone wrapper for indexing/search
│   └── vector\_store\_init.py       # (Optional) Pinecone index creation/init
├── example/
│   └── example.ipynb              # Notebook for experiments

````

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install openai arize openinference-instrumentation-openai arize-phoenix-evals pandas arize-otel
````

### 2. Add your credentials to `config.py`

```python
openai_api_key = "sk-..."
arizeai_api_key = "rz_..."
arizeai_proj_name = "My LLM Project"
arizeai_spid = "space_id_here"
pinecone_api_key = "pc_..."
pinecone_index_name = "index-policies"
```

---

## 🧠 What It Does

1. Takes a batch of prompts.
2. Retrieves relevant context chunks from Pinecone (RAG).
3. Feeds the context into the developed LLM to generate a response.
4. Evaluates each response using LLM-as-a-Judge QA evaluator.
5. Returns:

   * ✅ Prompt
   * 📚 Retrieved Reference
   * 🤖 Model Output
   * 🏷️ Label (`correct` / `incorrect`)
   * 📈 Score 
   * 🧾 Explanation

---

## ✅ Example Output

```
--- Prompt 1 ---
Prompt: How long do I have to request a refund?
Reference: Refunds must be requested within 30 days of the original transaction.
Model Output: Within 30 days of the original transaction.
Label: correct
Score: 1
Explanation: The answer exactly matches the reference text.
```

---

## 🧪 Testing

Run full unit tests:

```bash
python3 -m unittest test_main.py
```

### What’s tested:

* Structure and content of evaluation results
* Labels are valid (`CORRECT`, `INCORRECT`, `PARTIALLY_CORRECT`)
* Scores are within range (0.0 to 1.0)
* Batch output matches input count

---




## 🔍 Using Arize AI for Model Monitoring, Tracing & Evaluation

![Arize Dashboard](https://github.com/ArianFotouhi/llmPostDeployment-EvaluationAndMonitoring/blob/main/assets/ArizeAI_dashboard.png)
![Arize Monitoring](https://github.com/ArianFotouhi/llmPostDeployment-EvaluationAndMonitoring/blob/main/assets/ArizeAI_monitoring.png)
![Arize Tracibility](https://github.com/ArianFotouhi/llmPostDeployment-EvaluationAndMonitoring/blob/main/assets/ArizeAI_tracability.png)


**Arize AI** offers an end-to-end observability platform for machine learning models in production. It enables you to:

---

### 1. 📊 Monitor Model Health

You can track critical metrics like:

- **Token usage** (prompt, completion, total)
- **Latency** (P50, P99, etc.)
- **Custom thresholds** to detect drift or quality drops

Dashboards provide **hourly and daily trends** across model versions to detect anomalies, usage spikes, or latency issues. You can set up monitors that trigger alerts when values cross thresholds (e.g., `latency_ms > 5227`).

#### ⏱ What Is Latency (P50, P99, etc.)?

Latency metrics show how long the model takes to respond. Percentile-based latency gives insight into both average and worst-case performance:

- **P50 latency = 1.5 seconds** → Half of the responses are faster than 1.5s
- **P99 latency = 18 seconds** → 1 in 100 requests could take up to 18s, which may impact user experience

These are critical for identifying **performance bottlenecks** and **outlier behavior**.

---

### 2. 🔎 Trace LLM Activity

Every model interaction (trace) is logged with:

- Input/output data
- Token counts
- Response time
- Evaluation status (e.g., `"qa correct"`)

This allows you to **audit and debug** requests, trace performance regressions, and correlate usage patterns with output quality.

---

### 3. 🧪 Evaluate Responses at Scale

Arize supports both manual and automated evaluations through:

- **Eval pipelines** (e.g., tagging correct/incorrect completions)
- **Labeling queues**
- **Custom metrics** for output quality

This helps you maintain model accuracy and consistency even as your deployment scales.

---


