# 🎓 AI Placement Preparation Assistant

An AI-powered placement preparation assistant built using **Generative AI, Retrieval-Augmented Generation (RAG), LangChain, Google Gemini, FAISS, and Streamlit**.

This application allows users to upload placement preparation PDFs and ask questions about their content. It retrieves relevant information from the uploaded documents and generates answers with source references.

## 🚀 Features

- 📄 Upload multiple placement preparation PDFs.
- ✂️ Extract and split PDF text into manageable chunks.
- 🧠 Generate vector embeddings using Google Gemini.
- 🔍 Retrieve relevant information using FAISS similarity search.
- 🤖 Generate context-based answers using Google Gemini.
- 📚 Display source filenames and page numbers.
- 💻 Interactive web interface built with Streamlit.
- 🛡️ Designed to answer using uploaded material and indicate when information cannot be found.

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- Google Gemini API
- Google GenAI SDK
- FAISS
- PyPDF
- python-dotenv

## 🏗️ Project Workflow

1. Upload placement preparation PDF documents.
2. Extract text from the uploaded PDFs.
3. Split the text into chunks using `RecursiveCharacterTextSplitter`.
4. Generate embeddings for the chunks using Gemini.
5. Store the vectors in a FAISS vector database.
6. Retrieve the three most relevant chunks for a user's question.
7. Pass the retrieved context to Gemini to generate an answer.
8. Display the answer and source references in Streamlit.

## 📂 Project Structure

```text
Placement_preparation_ai/
│
├── app.py
├── .env
├── .gitignore
├── requirements.txt
└── data/
    └── QAE.pdf
```

The `data/` folder is optional if you upload PDFs directly through the application.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Placement-Preparation-Assistant.git
cd AI-Placement-Preparation-Assistant
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add:

```env
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

Replace the placeholder with your own Gemini API key. Keep `.env` private and never commit it to GitHub.

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## 💡 Example Questions

- Explain Python lists and tuples.
- What is normalization in DBMS?
- Explain SQL joins.
- What are the important aptitude topics?
- How should I prepare for HR interviews?

Answers depend on the information available in the uploaded PDFs.

## 🔐 Security

- Store API keys in environment variables.
- Never commit `.env` or other files containing secrets.
- Keep `.venv/` and Python cache files out of version control.

## 🔮 Future Enhancements

- Add interview simulation.
- Generate quizzes from uploaded documents.
- Add chat history.
- Support additional document formats.
- Add persistent vector storage.

## 👨‍💻 Author

**Your Name**

Connect with me on [LinkedIn](YOUR_LINKEDIN_URL).

---

⭐ If you find this project useful, consider giving the repository a star!