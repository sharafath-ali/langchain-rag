import { Ollama } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import path from "node:path";

class PdfQA {

  // Accept 2 more parameters for configuring the document splitting process:
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
    // Important parameters to know here are chunkSize and chunkOverlap. chunkSize controls the max size (in terms of number of characters) of the final documents. chunkOverlap specifies how much overlap there should be between chunks. This is often helpful to make sure that the text isn't split weirdly.
    // https://js.langchain.com/v0.1/docs/modules/data_connection/document_transformers/#get-started-with-text-splitters
    const textSplitter = new CharacterTextSplitter({ 
      separator: " ",
      chunkSize: this.chunkSize, // Chunk size is measured by the number of characters
      // See: https://chunkviz.up.railway.app/
      chunkOverlap: this.chunkOverlap 
    });
    this.texts = await textSplitter.splitDocuments(this.documents);

  }

}

const pdfDocument = "../materials/pycharm-documentation-mini.pdf";

const pdfQa = await new PdfQA({ 
  model: "llama3", 
  pdfDocument,
  chunkSize: 1000,
  chunkOverlap: 0 
}).init();

console.log(pdfQa.texts.length); // 14