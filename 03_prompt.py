import asyncio
from queue import Queue
from threading import Thread

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.callbacks import StreamingStdOutCallbackHandler, BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tenacity import sleep

from redundant_filter_retriever import RedundantFilterRetriever
import langchain

langchain.debug = False

load_dotenv()


class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue: Queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put(token)

    def on_llm_end(self, *args, **kwargs) -> None:
        # Signal that the generation is complete
        self.queue.put(None)


# class AsyncStreamingHandler(BaseCallbackHandler):
#     def __init__(self, queue: AsyncQueue):
#         self.queue = queue
#
#     async def on_llm_new_token(self, token: str, **kwargs) -> None:
#         await self.queue.put(token)
#
#     async def on_llm_end(self, *args, **kwargs) -> None:
#         # Signal that the generation is complete
#         await self.queue.put(None)
#

class QAChain:
    def __init__(self, chat_model, retriever):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer any use questions based solely on the context below:"
                       "<context>"
                       "{context}"
                       "</context>"),
            ("placeholder", "{chat_history}"),
            ("human", "Answer:"),
        ])

        self.combine_docs_chain = create_stuff_documents_chain(
            chat_model,
            self.prompt
        )

        self.chain = create_retrieval_chain(retriever, self.combine_docs_chain)

    def ask_question_sync(self, question: str) -> Queue:
        """
        Synchronous version that returns a Queue containing streaming tokens
        """
        queue = Queue()
        handler = StreamingHandler(queue)

        # Run in a separate thread to not block
        def run_chain():
            self.chain.invoke(
                {"input": question},
                config=RunnableConfig(callbacks=[handler])
            )

        import threading
        thread = threading.Thread(target=run_chain)
        thread.start()

        return queue

# Example usage for synchronous version
def demo_sync(qa_chain: QAChain, question: str):
    print("Starting synchronous demo...")
    queue = qa_chain.ask_question_sync(question)

    # Process tokens as they arrive
    while True:
        token = queue.get()
        if token is None:  # End of generation
            break
        sleep(0.1)
        print(token, end='', flush=True)
    print("\nSync demo complete!")


if __name__ == '__main__':
    # Initialize chat model

    # Initialize retriever
    embeddings = OpenAIEmbeddings()
    db = Chroma(
        persist_directory="emb",
        embedding_function=embeddings
    )

    # retriever = db.as_retriever()
    retriever = RedundantFilterRetriever(
        embeddings=embeddings,
        chroma=db
    )

    # Initialize the chain
    chat_model = ChatOpenAI(streaming=True)
    qa_chain = QAChain(chat_model, retriever)

    # Create and run the async event loop
    question ="What is an interesting fact about the English language?"
    demo_sync(qa_chain, question)