import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import path from "node:path";
// import { RetrievalQAChain } from "langchain/chains"; // Deprecated
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

class PdfQA {

  constructor({ model, pdfDocument, chunkSize, chunkOverlap, searchType = "similarity", kDocuments }) {

    this.model        = model;
    this.pdfDocument  = pdfDocument;
    this.chunkSize    = chunkSize;
    this.chunkOverlap = chunkOverlap;

    this.searchType   = searchType;
    this.kDocuments   = kDocuments;

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

    // Deprecated:
    // const chain = RetrievalQAChain.fromLLM(this.llm, this.retriever);
    // Replace with suggested code: https://v02.api.js.langchain.com/classes/langchain.chains.RetrievalQAChain.html

    // We are using a custom Chat Prompt Template for this one, so that the LLM has a context provided along with our query. The Prompt must contain input variable "context", which will be used for passing in the formatted documents to createStuffDocumentsChain below:
    const prompt = ChatPromptTemplate.fromTemplate(`Answer the user's question: {input} based on the following context {context}`);

    // Creates a chain that passes a list of documents to a model
    // Ref: https://v02.api.js.langchain.com/functions/langchain.chains_combine_documents.createStuffDocumentsChain.html
    // Source: https://github.com/langchain-ai/langchainjs/blob/3d2aabac1fa876ed908cd751d9afb448692ebbb3/langchain/src/chains/combine_documents/stuff.ts
    // Prompt requires a {context} variable for this to work.
    const combineDocsChain = await createStuffDocumentsChain({
      llm: this.llm,
      prompt,
    });

    // Create a retrieval chain that retrieves documents and then passes them on
    // Ref: https://v02.api.js.langchain.com/functions/langchain.chains_retrieval.createRetrievalChain.html
    // Source: https://github.com/langchain-ai/langchainjs/blob/3d2aabac1fa876ed908cd751d9afb448692ebbb3/langchain/src/chains/retrieval.ts#L73
    const chain = await createRetrievalChain({
      combineDocsChain,
      retriever: this.retriever,
    });

    // A rough overview of the final `chain` flow:
    // 1) chain gets triggered by `chain.invoke({ input: "... "})` 
    // 2) A retriever is set in place according to the value passed to the `retriever` property of `createRetrievalChain()`
    // 3) The retriever retrieves the relevant Document objects as an Array: [ Document {}, Document {}, Document {} ]
    // 4) This array of documents is passed through to the next chain link through a context variable: { context: [ Document {}, Document {}, ... ], input, ... }
    // 5) The next link is part of the combineDocsChain, which converts the array of documents found in the context property to a large string: { context: [ Document {}, Document {}, ...] } => { context: String }
    // 6) In this step, the prompt accepts the input along with the context from the previous steps and passes this to the LLM to answer the question: "Based on this context {context} answer this question {input}"
    // 7) The LLM step appends the answer to the object received and passes it on to the next step in the chain: { context: String, input, answer: "...", ...}
    // 8) The output of the chain has a property answer now that contains the LLM response.

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
  kDocuments: 5 
}).init();

const pdfQaChain = pdfQa.queryChain();

// Let's try it out by asking how we can debug in PyCharm.
const answer1 = await pdfQaChain.invoke({ input: "How do we add a custom file type in PyCharm?" });
console.log( "🤖", answer1.answer, "\n" );
console.log( "# of documents used as context: ", answer1.context.length, "\n" );
