import gradio as gr
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from trafilatura import fetch_url, extract
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline
import torch
import os
from dotenv import load_dotenv


load_dotenv()

access_token = os.getenv("CLI_TOKEN")

model_name_or_path = "mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-AWQ-calib-ja-100k"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                          trust_remote_code=True)
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, 
                                          safetensors=True, 
                                          fuse_layers=True)

if torch.cuda.is_available():
    model = model.to("cuda")

url = "https:##########################"
filename = 'textfile.txt'

document = fetch_url(url)
text = extract(document)

with open(filename, 'w', encoding='utf-8') as f:
    f.write(text)

loader = TextLoader(filename, encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator = "\n",
    chunk_size=200,
    chunk_overlap=20,
)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "参考情報を元に、ユーザーの質問にできるだけ正確に答えてください。"
text = "{context}\nユーザの質問は、次のとおりです。\n{question}"

template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    batch_size=1, 
)

PROMPT = PromptTemplate(
    template=template,
    input_variables=["query"],  
    template_format="f-string"
)

chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs=dict(temperature=0.1, 
                          do_sample=True, 
                          repetition_penalty=1.1)
    ),
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    verbose=True,
)

def generate_response(user_inputs, history):
    query = user_inputs
    result = qa({"query": query})
    return result["result"]

iface = gr.ChatInterface(fn=generate_response, 
                     title="RAG-based Q&A",
                     description="ELYZAに何でも質問してみてください。",
                     theme="soft",
                     retry_btn=None,
                     undo_btn="Delete Previous",
                     clear_btn="Clear",
                    )

iface.launch()
