import os
import gradio as gr
import pdfplumber
import chromadb
from google import genai
from google.genai import types
from google.api_core import retry
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Annotated, List, Optional, Dict

# ==== Setup ====

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key is missing. Set it as a secret in Hugging Face.")

# Initialize Google Gemini Client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Retry condition
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


# ==== Embedding Function ====

class GeminiEmbeddingFunction:
    def __init__(self, document_mode=True):
        self.document_mode = document_mode

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input):
        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]


# ==== PDF Extraction ====

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return text.strip()


# ==== ChromaDB Setup ====

DB_NAME = "document_db"
embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True
chroma_client = chromadb.PersistentClient(path="./chroma_db")
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)


# ==== LangGraph State ====

class QAState(TypedDict):
    user_question: str
    uploaded_doc_text: Optional[str]
    conversation_history: List[Dict[str, str]]
    response: Optional[str]

def check_doc_uploaded(state: QAState) -> str:
    return "answer" if state.get("uploaded_doc_text") else "no_doc"

def no_doc_response(state: QAState) -> QAState:
    return {
        **state,
        "response": (
            "📄 Hi! I'm a Teaching Assistant that answers questions based **only** on the uploaded PDF.\n\n"
            "Please upload a document so I can help you with specific answers."
        )
    }

def answer_with_doc(state: QAState) -> QAState:
    question = state["user_question"]
    document = state["uploaded_doc_text"]
    history = state["conversation_history"]

    if not question.strip():
        return {**state, "response": "❓ Please enter a question."}

    embed_fn.document_mode = False
    result = db.query(query_texts=[question], n_results=1)
    retrieved_texts = result["documents"][0] if "documents" in result else []

    if not retrieved_texts:
        return {**state, "response": "🤔 I couldn't find relevant information in the document."}

    # Add to prompt before QUESTION:
    history_prompt = ""
    if history:
        history_prompt = "Conversation so far:\n" + "\n".join(
            [f"Q: {q['question']}\nA: {q['answer']}" for q in history[-5:]]
        )



    prompt = f"""
    "system",
    "You are an AI-powered Teaching Assistant designed to help users learn from documents they upload. "
    "You will analyze the uploaded document, extract the core topic or subject, and use that as the basis for teaching. "
    "\n\n"
    "Once the topic is extracted, you will guide the user through interactive learning using different modes:\n"
    "- In 'lesson' mode: Explain the topic in depth using simple language and relevant examples.\n"
    "- In 'quiz' mode: Ask short, concept-reinforcing questions based on the document content.(Ask 5 questions at a go both objective and subjective)\n"
    "Adapt to the user’s learning pace, encourage curiosity, and maintain a friendly and supportive tone. "
    "In whatever mode you are in you give the user, your student proper explanation even if its just an answer, you say why it is wrong or why it is correct. Once they complete one topic you say lets go to the next topic"
    "If they are in test mode after they answer a question you say if its write or wrong with explanation"
    "Wrap up the session politely when finished."
    "If the user says lets start with Chapter 1 it means you will teach chapter 1 unless explicitly specified as quiz or test you dont ask questions but if you have completed teaching the topic you politely ask if we can test their knowledge on what they have learnt"
    You also remember the previous conversation and can summarize or refer back when asked.
    {history_prompt}
    QUESTION: {question}
    """
    for passage in retrieved_texts:
        prompt += f"\nPASSAGE: {passage}"

    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)

    new_history = history + [{"question": question, "answer": response.text}]
    return {
        **state,
        "response": response.text if response else "❌ I couldn't generate an answer.",
        "conversation_history": new_history
    }

def build_qa_graph():
    graph = StateGraph(QAState)
    graph.add_node("no_doc", RunnableLambda(no_doc_response))
    graph.add_node("answer", RunnableLambda(answer_with_doc))
    graph.set_conditional_entry_point(check_doc_uploaded, {
        "no_doc": "no_doc",
        "answer": "answer"
    })
    graph.add_edge("no_doc", END)
    graph.add_edge("answer", END)
    return graph.compile()

qa_graph = build_qa_graph()


# ==== Gradio UI ====

with gr.Blocks() as app:
    gr.Markdown("## 📄 AI PDF Q&A with Google Gemini + LangGraph")
    gr.Markdown("Upload a PDF and ask questions based only on its contents!")

    embeddings_ready = gr.State(False)
    uploaded_doc_text = gr.State("")
    conversation_history = gr.State([])

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", type="filepath")
        upload_status = gr.Textbox(label="Upload Status", interactive=False)

    def upload_pdf(pdf_file):
        if pdf_file is None:
            return "❌ Please upload a PDF.", False, "", []

        # Extract and embed
        pdf_text = extract_text_from_pdf(pdf_file)
        db.delete(ids=["doc1"])
        db.add(documents=[pdf_text], ids=["doc1"])
        return "✅ PDF uploaded successfully!", True, pdf_text, []

    upload_button = gr.Button("Upload & Store Embeddings")
    upload_button.click(
        fn=upload_pdf,
        inputs=pdf_input,
        outputs=[upload_status, embeddings_ready, uploaded_doc_text, conversation_history]
    )

    with gr.Row():
        question_input = gr.Textbox(label="Ask a Question", placeholder="e.g. What are the steps to apply?")
        answer_output = gr.Textbox(label="Answer", interactive=False)

    def ask_question_fn(user_question, doc_text, history, embeddings_ready):
        state = {
            "user_question": user_question,
            "uploaded_doc_text": doc_text if embeddings_ready else None,
            "conversation_history": history
        }
        new_state = qa_graph.invoke(state)
        return new_state["response"], new_state["conversation_history"]

    ask_button = gr.Button("Get Answer")
    ask_button.click(
        fn=ask_question_fn,
        inputs=[question_input, uploaded_doc_text, conversation_history, embeddings_ready],
        outputs=[answer_output, conversation_history]
    )

app.launch()
