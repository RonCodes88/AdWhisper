from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from chroma import ChromaDB

app = FastAPI()
db = ChromaDB()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to AdWhisper API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/documents") # sample endpoint to retrieve ChromaDB documents
async def get_documents():
    documents = db.collection.get_all()
    return {"documents": documents}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    

