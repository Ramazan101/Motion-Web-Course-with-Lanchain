import os
import uvicorn
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette import status

# Document(
#     page_content="Academy Motion Web - Programming course and design",
#     metadata={"source": "text.txt"}
# )

loaders = TextLoader("text.txt", encoding="utf-8")
document = loaders.load()

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=1000,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_over=20,
)

chunks = text_splitter.split_documents(document)

embeddings = ?(
    model="?"
)

InMemoryVectorStore.from_documents(chunks, embeddings)

chat_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "Ты онлайн менеджер Motion Web IT Academy",
            "Должен отвечать на вопросы четко и коротко",
            "При ответе ты должен сперва прочитать Базу Знаний",
            "База Знаний {base_knowledge}",
            "Если человек написал то чего нету в базе ты должен ответить что такая инфа у нас нету НО только на своем"
        ),
        "human",
        "{content}"
    ]
)


chain = chat_prompt | llm | StrOutputParser()


test_app = FastAPI()

class QuestionSchema(BaseModel):
    question: str

@test_app.post("/questions/")
async def answer(data: QuestionSchema):
    data = data.question.strip()

    if not data:
        raise HTTPException(status_code=400, detail="Information not correct!")

    new_document = vector_store.asimilarity_search(data, k=2)
    final_answer = chain.invoke({"content": data, "base_knowledge": new_document})
    return {"answer": final_answer}

if __name__ == "__main__":
    uvicorn.run("main:test_app", host="127.0.0.1", port=8000)