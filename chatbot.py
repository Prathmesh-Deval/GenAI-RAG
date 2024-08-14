import gradio as gr
import fitz
import os
from PIL import Image
import pytesseract
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
import json

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'


def process_pdf(pdf_file):
    pdf_path = pdf_file.name
    output_image_dir = 'images'
    os.makedirs(output_image_dir, exist_ok=True)

    pdf_document = fitz.open(pdf_path)

    text = ""
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        output_image_path = os.path.join(output_image_dir, f'page_{page_number}.png')
        pix.save(output_image_path)

        image = Image.open(output_image_path)
        text += pytesseract.image_to_string(image)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=30)
    texts = text_splitter.split_text(text)

    EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": "cpu"})

    docsearch = FAISS.from_texts(texts, embeddings)

    docsearch.save_local("faiss_store/","index1")

    return "Embeddings created. You can now ask questions."


def query_knowledge_base(question):
    EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": "cpu"})

    docsearch = FAISS.load_local(folder_path="faiss_store/", embeddings=embeddings, index_name="index1",allow_dangerous_deserialization=True)


    docs = docsearch.similarity_search(query=question, k=8)

    parameters = {
        "max_new_tokens": 150,
        "num_return_sequences": 1,
        "top_k": 10,
        "top_p": 0.95,
        "do_sample": False,
        "return_full_text": False,
        "temperature": 0.1
    }

    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
            input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json[0]["generated_text"]

    content_handler = ContentHandler()
    sm_llm_mistral_instruct = SagemakerEndpoint(
        endpoint_name="pdf-endpoint",
        region_name="ap-south-1",
        model_kwargs=parameters,
        content_handler=content_handler,
    )

    prompt_template = """system

    This is a conversation between an AI assistant and a Human.

    user

    Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    #### Context ####
    {context}
    #### End of Context ####

    Question: {question}
    assistant

    Answer:""".strip()

    sm_llm_mistral_instruct.model_kwargs = {
        "max_new_tokens": 300,
        "num_return_sequences": 1,
        "top_k": 10,
        "top_p": 0.99,
        "do_sample": False,
        "return_full_text": False,
        "temperature": 0.1,
    }

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=sm_llm_mistral_instruct, prompt=PROMPT)

    result_mistral = chain({"input_documents": docs, "question": question}, return_only_outputs=False)["output_text"]

    return result_mistral


with gr.Blocks() as demo:
    with gr.Tab("Upload PDF"):
        pdf_upload = gr.File(label="Upload PDF", type="filepath")
        submit_button = gr.Button("Create Embeddings")
        message = gr.Textbox(label="Status", interactive=False)

        submit_button.click(fn=process_pdf, inputs=pdf_upload, outputs=message)

    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(label="Enter your question")
        answer_output = gr.Textbox(label="Answer", interactive=False)

        question_input.submit(fn=query_knowledge_base, inputs=question_input, outputs=answer_output)

if __name__ == "__main__":
    demo.launch()

