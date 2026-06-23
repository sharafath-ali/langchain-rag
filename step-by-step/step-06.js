import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import path from "node:path";

class PdfQA {

  constructor({ model, pdfDocument, chunkSize, chunkOverlap }) {

    this.model        = model;
    this.pdfDocument  = pdfDocument;
    this.chunkSize    = chunkSize;
    this.chunkOverlap = chunkOverlap;

  }

  async init(){
    this.initChatModel();
    await this.loadDocuments();
    await this.splitDocuments();
    // Load the Ollama model that will be used to convert the document text into embeddings:
    this.selectEmbedding = new OllamaEmbeddings({ model: "all-minilm:latest" });
    // Initialize our Vector store:
    await this.createVectorStore();
    return this;
  }

  initChatModel(){
    console.log("Loading model...");
    this.llm = new Ollama({ model: this.model });
  }

  async loadDocuments(){
    console.log("Loading PDFs...");
    const pdfLoader = new PDFLoader(path.join(import.meta.dirname,this.pdfDocument));
    this.documents = await pdfLoader.load();
  }

  async splitDocuments(){
    console.log("Splitting documents...");
    const textSplitter = new CharacterTextSplitter({ 
      separator: " ",
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap 
    });
    this.texts = await textSplitter.splitDocuments(this.documents);

  }

  async createVectorStore(){
    console.log("Creating document embeddings...");
    // Use an In-memory Vector store provided by LangChain to store the embedding vectors
    // You can replace this vector store with an actual Vector Database such as Pinecode, ChromaDB, Supabase and other. See: https://js.langchain.com/v0.1/docs/modules/data_connection/vectorstores/
    this.db = await MemoryVectorStore.fromDocuments(this.texts, this.selectEmbedding);
  }

}

const pdfDocument = "../materials/pycharm-documentation-mini.pdf";

const pdfQa = await new PdfQA({ 
  model: "llama3", 
  pdfDocument,
  chunkSize: 1000,
  chunkOverlap: 0 
}).init();

console.log( "Embeddings model: ", pdfQa.db.embeddings.model ); // all-minilm:latest
console.log( "# of embeddings: ", pdfQa.db.memoryVectors.length ); // 14 

// Query the Vector store directly: https://js.langchain.com/v0.2/docs/integrations/vectorstores/memory/#query-directly
const similaritySearchResults = await pdfQa.db.similaritySearch("File type associations",2);
console.log("\nDocument pages related to our query:");
for (const doc of similaritySearchResults) {
  console.log(`\n* ${JSON.stringify(doc.metadata.loc, null)}\n`);
  // console.log(doc.pageContent);
}

// If you want to execute a similarity search and receive the corresponding scores you can run:
const similaritySearchWithScoreResults = await pdfQa.db.similaritySearchWithScore("File type associations", 10);

console.log("Document pages and their score related to our query:");
for (const [doc, score] of similaritySearchWithScoreResults) {
  console.log(`* [SIM=${score.toFixed(3)}] [Page number: ${doc.metadata.loc.pageNumber}]`);
}
