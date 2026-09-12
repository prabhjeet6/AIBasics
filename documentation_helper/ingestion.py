import asyncio

from typing import  List



from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl,TavilyExtract,TavilyMap

from config import  settings
from ragutils import get_embeddings_model, get_vector_store



embeddings=get_embeddings_model()
vectorstore=get_vector_store(settings.langchain_documentation_index_name)

tavily_extract=TavilyExtract(max_depth=5,max_breadth=20,max_pages=1000)
tavily_map=TavilyMap()
tavily_crawl=TavilyCrawl()

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    print("VECTOR STORAGE PHASE")
    print(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",

    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    print(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            print(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            print(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        print(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        print(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    print("DOCUMENTATION INGESTION PIPELINE")

    print(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site"

    )
    # Crawl the documentation site

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        print(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Split documents into chunks
    print("DOCUMENT CHUNKING PHASE")
    print(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap"

    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    print(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    print("PIPELINE COMPLETE")
    print("🎉 Documentation ingestion pipeline finished successfully!")
    print("📊 Summary:")
    print(f"   • Documents extracted: {len(all_docs)}")
    print(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
    print("Ingestion Ends")