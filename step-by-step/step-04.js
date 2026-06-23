import { Ollama } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import path from "node:path";

class PdfQA {

  // We need to add a new parameter (pdfDocument) to pass the PDF document pathname:
  constructor({ model, pdfDocument }) {
    this.model = model;
    this.pdfDocument = pdfDocument;
  }

  // We need to make this method async in order to use await inside the function body
  async init(){
    this.initChatModel();
    // Await for the provided document to be loaded:
    await this.loadDocuments();
    return this;
  }

  initChatModel(){
    console.log("Loading model...");
    this.llm = new Ollama({ model: this.model });
  }

  async loadDocuments(){
    console.log("Loading PDFs...");
    // Use the PDFLoader component provided by LangChain to load the PDF
    // Reference: https://js.langchain.com/v0.2/docs/integrations/document_loaders/file_loaders/pdf/

    const pdfLoader = new PDFLoader(path.join(import.meta.dirname,this.pdfDocument));
    // By default, PDFLoader will create a Document object for each page. You can set the `splitPages` parameter to false to have one Document per PDF:
    // new PDFLoader( <PATH>, { splitPages: false });
    this.documents = await pdfLoader.load();
  }

}

const pdfDocument = "../materials/pycharm-documentation-mini.pdf";
const pdfQa = await new PdfQA({ model: "llama3", pdfDocument }).init();
// The number of Documents created by the PDFLoader, as it splits the PDF into multiple documents if the splitPages is not set to false
console.log( pdfQa.documents.length ); // 9

// The Document text content:
console.log( "\n\nDocument #0 page content: ", pdfQa.documents[0].pageContent );

// The Document metadata:
console.log( "\n\n Document #0 metadata: ", pdfQa.documents[0].metadata );
