import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loaders = TextLoader("text.txt", encoding="utf-8")
document = loaders.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
)
chunks = text_splitter.split_documents(document)


embeddings = FastEmbedEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=1000,
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты онлайн менеджер Motion Web IT Academy.\n"
            "Должен отвечать на вопросы четко и коротко.\n"
            "При ответе ты должен сперва прочитать Базу Знаний.\n"
            "База Знаний:\n{base_knowledge}\n\n"
            "Если человек спросил то, чего нет в базе, ответь своими словами, что такой информации у нас нет.",
        ),
        ("human", "{content}"),
    ]
)

chain = chat_prompt | llm | StrOutputParser()

test_app = FastAPI(title="Motion Web QA Bot")


class QuestionSchema(BaseModel):
    question: str


@test_app.post("/questions/")
async def answer(data: QuestionSchema):
    question_text = data.question.strip()

    if not question_text:
        raise HTTPException(status_code=400, detail="Information not correct!")

    docs = await vector_store.asimilarity_search(question_text, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_answer = await chain.ainvoke(
        {"content": question_text, "base_knowledge": context}
    )
    return {"answer": final_answer}


if __name__ == "__main__":
    uvicorn.run("main:test_app", host="127.0.0.1", port=8000, reload=True)