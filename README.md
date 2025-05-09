# 📄 PDF TA: AI-Powered Teaching Assistant for PDFs

PDF TA is an AI-powered teaching assistant that allows users to upload any PDF document and interact with its content through intelligent Q&A. It uses **Google Gemini** for embeddings and content generation, **ChromaDB** for semantic retrieval, and **LangGraph** for dynamic reasoning — all wrapped in an intuitive **Gradio UI**.

## 🚀 Features

- ✅ Upload any educational or instructional PDF
- 💬 Ask context-aware questions based *only* on the uploaded document
- 📚 Intelligent retrieval of document sections using embeddings
- 🎓 AI-guided teaching, quiz, and test support with detailed explanations
- 🧠 Maintains and refers to past conversation history
- ⚙️ Built-in logic to adapt to different learning modes (lesson, quiz, test)

## 🛠️ Tech Stack

- [Google Gemini API](https://ai.google.dev) (Text Embedding + Content Generation)
- [ChromaDB](https://www.trychroma.com/) (Persistent Vector Database)
- [LangGraph](https://github.com/langchain-ai/langgraph) (Conversational State Management)
- [Gradio](https://www.gradio.app) (Web Interface)
- [pdfplumber](https://github.com/jsvine/pdfplumber) (PDF Text Extraction)

## 📦 Installation

```bash
git clone https://github.com/Sharanya-krishnamurthi/PdfTA.git
cd PdfTA
pip install -r requirements.txt
````

Make sure to add your Google API key as an environment variable:

```bash
export GOOGLE_API_KEY="your-google-api-key"
```

## 🧪 Running the App

```bash
python app.py
```

It will launch a local Gradio interface where you can upload a PDF and start interacting with it.

## 🔍 Example Use Cases

* 📘 Learning from lecture notes or academic PDFs
* 🧾 Understanding legal or instructional documents
* 🎯 Self-paced teaching and concept reinforcement
* 🎓 AI teaching assistant for classrooms or individual learners

## 💡 Future Enhancements

* PDF section navigation
* Multi-PDF comparison
* Voice interaction using TTS/STT
* User authentication and history tracking

## 📄 License

MIT License

---

Created with ❤️ using Google Gemini + LangGraph


## DEMO
[HuggingFace Spaces](https://huggingface.co/spaces/sharanya/TeachingAgent)
