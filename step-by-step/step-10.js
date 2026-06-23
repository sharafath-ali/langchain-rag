import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import path from "node:path";
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
  kDocuments: 5 
}).init();

const pdfQaChain = pdfQa.queryChain();

// Let's try it out by asking how we can debug in PyCharm.
const answer1 = await pdfQaChain.invoke({ input: "How do we add a custom file type in PyCharm?" });
// console.log( "🤖", answer1.answer, "\n" );
console.log( "# of documents used as context: ", answer1.context.length, "\n" );

// Let's have a look at the information it was based on from the documentation
for ( const document of answer1.context ){
  const index = answer1.context.findIndex(doc => doc === document);
  console.log(`\n 📄 DOCUMENT: No${index+1}`);
  console.log(document.pageContent.trim(), "\n");
}

// We store the previous question along with its answer in an array:
const chatHistory1 = [ answer1.input, answer1.answer ];
// The createRetrievalChain part of the chain, allows us to pass an optional chat_history argument containing extra context. This will be used to give us a more relevant result since it acts as a kind of short memory to our LLM  
const answer2 = await pdfQaChain.invoke({ 
  input: "Is there anything more to add here?", 
  chat_history: chatHistory1 
});

console.log( "🤖", answer2.answer, "\n" );

// Original question:
// const question3 = "Wie kann man PyCharm installieren?"; 
// Question tailored for mini version of the documentation:
const question3 = "Wie fügt man einen benutzerdefinierten Dateityp in PyCharm hinzu?";
const answer3 = await pdfQaChain.invoke({ input: question3 });

// ...and get a relevant answer in German!
console.log( "🤖", question3, "\n", answer3.answer, "\n" );