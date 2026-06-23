import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import path from "node:path";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

class PdfQA {

  // Add an extra temperature parameter for setting a custom temperature for the model
  constructor({ model, pdfDocument, chunkSize, chunkOverlap, searchType = "similarity", kDocuments, temperature = 0.8 }) {

    this.model        = model;
    this.pdfDocument  = pdfDocument;
    this.chunkSize    = chunkSize;
    this.chunkOverlap = chunkOverlap;
    this.searchType   = searchType;
    this.kDocuments   = kDocuments;
    this.temperature  = temperature;

  }

  async init(){
    this.initChatModel();
    await this.loadDocuments();
    await this.splitDocuments();
    this.selectEmbedding = new OllamaEmbeddings({ model: "all-minilm:latest" });
    await this.createVectorStore();
    this.createRetriever();
    this.chain = await this.createChain();
    return this;
  }

  initChatModel(){
    console.log("Loading model...");
    this.llm = new Ollama({ 
      model: this.model,
      // We are passing either the default (0.8) or custom temperature value here:
      temperature: this.temperature 
    });
  }

  async loadDocuments(){
    console.log("Loading PDFs...");
    const pdfLoader = new PDFLoader(path.join(import.meta.dirname,this.pdfDocument));
    this.documents = await pdfLoader.load();
  }

  async splitDocuments(){
    console.log("Splitting documents...");
    // Check this demo to learn more about how text splitters work: https://chunkviz.up.railway.app/
    const textSplitter = new RecursiveCharacterTextSplitter({ 
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap 
    });
    this.texts = await textSplitter.splitDocuments(this.documents);

  }

  async createVectorStore(){
    console.log("Creating document embeddings...");
    this.db = await MemoryVectorStore.fromDocuments(this.texts, this.selectEmbedding);
  }

  createRetriever(){
    console.log("Initialize vector store retriever...");
    this.retriever = this.db.asRetriever({ 
      k: this.kDocuments,
      searchType: this.searchType 
    });
  }

  async createChain(){
    console.log("Creating Retrieval QA Chain...");

    const prompt = ChatPromptTemplate.fromTemplate(`Answer the user's question: {input} based on the following context {context}`);

    const combineDocsChain = await createStuffDocumentsChain({
      llm: this.llm,
      prompt,
    });

    const chain = await createRetrievalChain({
      combineDocsChain,
      retriever: this.retriever,
    });

    return chain;
  }

  queryChain(){
    return this.chain;
  }

}

const pdfDocument = "../materials/pycharm-documentation-mini.pdf";

const pdfQa = await new PdfQA({ 
  model: "llama3", 
  pdfDocument,
  chunkSize: 1000,
  chunkOverlap: 0,
  searchType: "similarity",
  kDocuments: 5,
  // This time we can pass an extra parameter for the model's temperature
  // Learn more about temperature here: https://www.youtube.com/watch?v=q4ks_aUgGT4
  temperature: 0 // 0 is the strictest mode
}).init();

const pdfQaChain = pdfQa.queryChain();

const answer1 = await pdfQaChain.invoke({ input: "How do we add a custom file type in PyCharm?" });
console.log( "🤖", answer1.answer, "\n" );
// console.log( "# of documents used as context: ", answer1.context.length, "\n" );

