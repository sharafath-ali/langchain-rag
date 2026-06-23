import { Ollama } from "@langchain/ollama";

class PdfQA {

  // Accept one parameter for the moment (model)
  constructor({ model }) {
    // Add the argument passed to model to our class property model
    this.model = model;
  }

  // Special initializer method
  init(){
    // Load the LLM
    this.initChatModel();
    // Return this to retain access to the main object after calling init()
    return this;
  }

  initChatModel(){
    console.log("Loading model...");
    // Use the Ollama LangChain class to load an LLM
    this.llm = new Ollama({ model: this.model });
  }

}

const pdfQa = new PdfQA({ model: "llama3" }).init();

console.log({ pdfQa });