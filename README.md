Dzongkha AI Assistant 
A bilingual English–Dzongkha AI assistant built with Streamlit. The project combines a local AI chat workflow, Dzongkha language processing, optional live web retrieval, and document summarisation and translation in a Bhutanese-inspired interface.
Project status: Active development. AI-generated facts and machine translations require checking; free web search is best-effort and may fail or return incomplete results.

Features
- AI Chat: Ask questions in English or Dzongkha, with an English-to-Dzongkha response pipeline.
- Fine-tuned chat mode: Separate mode for a configured Dzongkha-capable model (requires compatible local model files).
- Live web retrieval: Attempts to fetch recent information for time-sensitive questions and displays source links when available. Sources are search excerpts, not independently verified facts.
- Document Studio: Upload text-based PDF, DOCX, TXT, or Markdown files to summarise or translate content between English and Dzongkha.
- Summarisation and next-word prediction: Dedicated tools accessible from the sidebar.
- Dzongkha text processing: Normalisation, spelling checks and sentence-structure warnings through dzongkha_nlp.py.
- Chat history: Local conversation management.
- Responsive UI: Bhutanese-inspired design with a welcome screen and traditional background artwork.
Requirements
- Python 3.10+ recommended
- macOS for the current MLX-based local inference setup; model support depends on your hardware and configuration
- Internet access for optional live search and any externally downloaded models
Setup
Clone the repository:
git clone https://github.com/03240393gcbs-creator/Dzo_chatbot.git
cd Dzo_chatbot
Create and activate a virtual environment (macOS):
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ddgs certifi pypdf python-docx
The final command adds packages used by the latest web-search and document-upload features. Model dependencies and any local model paths must also be configured as required by your installation.

Run the latest interface:
python -m streamlit run dzongkha_app_v3.py
Open the local URL printed in Terminal (usually http://localhost:8501). The repository also contains an older app.py entry point; use dzongkha_app_v3.py for the latest interface.
Using Document Studio
1. Open Summarise from the sidebar.
2. Upload a PDF, Word, TXT or Markdown file, or paste text.
3. Select Summarise or Translate.
4. Choose the available translation direction and generate your result.
5. Review the output and download it when offered.
Limits: Uploads are limited to 12 MB, and only the first 24,000 extracted characters are processed. Scanned/image-only PDFs are not supported without OCR. Long documents and translations may take time on local hardware.
Live information and reliability
The assistant attempts web retrieval for changing facts, such as current officeholders or news. If reliable sources cannot be retrieved, it should say that it cannot verify the current answer rather than guess. Search results can be stale or incorrect, so check linked primary sources for important claims. Local verified knowledge-bank entries may be used for non-current questions.
Main project files
File	Purpose
dzongkha_app_v3.py	Current Streamlit application
dzongkha_nlp.py	Dzongkha normalisation and language checks
verified_kb.json	Hand-checked answers for supported questions
requirements.txt	Base Python dependencies
welcome_bg.jpg	Welcome screen image (add to project root if needed)
app.py, dzongkha_app_v2.py	Earlier application versions
data/	Supporting language resources


Testing
If the project's test dependencies are installed:
python -m pytest -q
Privacy and security
Do not commit API keys, credentials, private chat histories, uploaded personal documents, .venv/, model weights or local caches. Check .gitignore before pushing. Uploaded content and search queries may be handled differently depending on the configured models and search providers.
Known limitations
- Machine-generated Dzongkha may contain spelling or grammatical errors.
- Free search providers may rate-limit or block requests.
- Web source links are not proof of independent fact verification.
- Document extraction supports text-based files, not scanned PDFs.
- MLX inference and translation speed depend on local hardware and models.
Repository
03240393gcbs-creator/Dzo_chatbot
Acknowledgements
Developed as a Dzongkha NLP/chatbot project to explore AI-assisted communication in Bhutan's national language.